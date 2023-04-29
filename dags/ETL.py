import os
import datetime
import csv
import json

from airflow.decorators import dag, task
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from transformers import pipeline
import pandas as pd


specialities_normalizer = json.load(
    open(
        os.path.join(os.path.dirname(__file__), "../data/specialities_normalizer.json"),
        "r",
    )
)
cities_normalizer = json.load(
    open(
        os.path.join(os.path.dirname(__file__), "../data/cities_normalizer.json"),
        "r",
    )
)


@task()
def extract_csv_data_task(**context):
    csv_path = context["params"]["csv_path"]

    with open(csv_path, mode="r", encoding="utf-8-sig") as csv_file:
        csv_reader = csv.DictReader(csv_file)
        csv_data = list(csv_reader)

    return csv_data


@task()
def extract_freetext_data_task():
    base_dir = "./data/free-text/"
    reports_files = os.listdir(base_dir)

    reports_files.remove("indexes")

    reports_records = []

    freetext_model = pipeline("ner", model="./models/freetext/")

    attribute_mapper = {
        "ProviderName": "first_name",
        "Speciality": "speciality",
        "City": "city",
        "Cost": "cost",
    }

    for report_file in reports_files[:3]:
        with open(base_dir + report_file, mode="r", encoding="utf-8-sig") as file:
            report = file.read()
            free_text_reports = report.splitlines()

            for free_text_report in free_text_reports:
                report_tokens_predictions = freetext_model(free_text_report)
                record = {
                    "first_name": "",
                    "last_name": "",
                    "speciality": "",
                    "city": "",
                    "working_hours": "",
                    "cost": "",
                }

                for prediction in report_tokens_predictions:
                    if prediction["entity"] == "Other":
                        continue

                    record[attribute_mapper[prediction["entity"]]] += (
                        prediction["word"]
                        if not record[attribute_mapper[prediction["entity"]]]
                        else prediction["word"].replace("##", "")
                        if prediction["word"].startswith("##")
                        else f" {prediction['word']}"
                    )

                if not not "".join(record.values()):
                    record["provider_network"] = "FreeText"
                    reports_records.append(record)

    return reports_records


# TODO: consider making transformation logic generic for similar data sources
@task()
def transform_vendor1_data_task(**context):
    csv_data = context["task_instance"].xcom_pull(task_ids="extract_csv_data_task")

    df = pd.DataFrame(csv_data)

    df["ProviderName"] = df["ProviderName"].str.replace(" ", "")
    df["ProviderName"] = (
        df["ProviderName"].str.replace("([A-Z][a-z]+)", " \g<1>").str.strip()
    )
    df["first_name"] = df["ProviderName"].str.split(" ", n=1, expand=True)[0]
    df["last_name"] = df["ProviderName"].str.split(" ", n=1, expand=True)[1]
    df["working_hours"] = df["End Hour"].astype(int) - df["Start Hour"].astype(int)
    df["Cost"] = df["Cost"].astype(float)
    df["provider_network"] = "Vendor1"

    for mapper in specialities_normalizer:
        df["Specialty"] = df["Specialty"].replace(
            mapper["specialities_cluster"], mapper["speciality"]
        )

    df["City"].apply(lambda x: x.replace("\\N", ","))
    for originalCity, normalizedCity in cities_normalizer.items():
        if normalizedCity:
            df["City"] = df["City"].replace(originalCity, normalizedCity)

    df = df.drop(columns=["ProviderName", "Start Hour", "End Hour"])

    df.rename(
        columns={
            "Specialty": "speciality",
            "City": "city",
            "Cost": "cost",
        },
        inplace=True,
    )

    return df.to_dict("records")


@task()
def transform_vendor2_data_task(**context):
    csv_data = context["task_instance"].xcom_pull(task_ids="extract_csv_data_task__1")

    df = pd.DataFrame(csv_data)

    df["first_name"] = df["Fname"].str.replace(" ", "")
    df["last_name"] = df["Lname"].str.replace(" ", "")
    df["provider_network"] = "Vendor2"
    df["working_hours"] = df["e_hour"].astype(int) - df["s_hour"].astype(int)
    df["cost"] = df["cost"].astype(float)

    for mapper in specialities_normalizer:
        df["spec"] = df["spec"].replace(
            mapper["specialities_cluster"], mapper["speciality"]
        )

    df["city"].apply(lambda x: x.replace("\\N", ","))
    for originalCity, normalizedCity in cities_normalizer.items():
        if normalizedCity:
            df["city"] = df["city"].replace(originalCity, normalizedCity)

    df = df.drop(columns=["Fname", "Lname", "s_hour", "e_hour"])

    df.rename(
        columns={"spec": "speciality"},
        inplace=True,
    )

    return df.to_dict("records")


@task()
def transform_oltp_data_task(**context):
    rows = context["task_instance"].xcom_pull(task_ids="extract_oltp_data_task")

    df = pd.DataFrame(
        rows,
        columns=[
            "first_name",
            "last_name",
            "speciality",
            "city",
            "provider_network",
            "working_hours",
            "cost",
        ],
    )

    df["working_hours"] = df["working_hours"].str.replace("hours", "").astype(int)
    df["cost"] = df["cost"].str.replace("$", "").astype(int)

    return df.to_dict("records")


@task()
def join_data_task(**context):
    [vendor1_data, vendor2_data, oltp_data, freetext_data] = context[
        "task_instance"
    ].xcom_pull(
        task_ids=[
            "transform_vendor1_data_task",
            "transform_vendor2_data_task",
            "transform_oltp_data_task",
            "extract_freetext_data_task",
        ]
    )

    return vendor1_data + vendor2_data + oltp_data + freetext_data


@task()
def load_data_task(**context):
    data = context["task_instance"].xcom_pull(task_ids="join_data_task")

    df = pd.DataFrame(data)
    unique_specialities = df["speciality"].unique()
    unique_cities = df["city"].unique()
    unique_provider_networks = df["provider_network"].unique()

    psql_conn = PostgresHook(postgres_conn_id="postgres_data_mart").get_conn()
    cursor = psql_conn.cursor()

    for speciality in unique_specialities:
        speciality = speciality.replace("'", "''")

        cursor.execute(
            f"""
            INSERT INTO "specialities" ("name")
            VALUES ('{speciality}')
            ON CONFLICT ("name") DO NOTHING;
            """
        )

    for city in unique_cities:
        city = city.replace("'", "''")

        cursor.execute(
            f"""
            INSERT INTO "cities" ("name")
            VALUES ('{city}')
            ON CONFLICT ("name") DO NOTHING;
            """
        )

    for provider_network in unique_provider_networks:
        provider_network = provider_network.replace("'", "''")

        cursor.execute(
            f"""
            INSERT INTO "provider_networks" ("name")
            VALUES ('{provider_network}')
            ON CONFLICT ("name") DO NOTHING;
            """
        )

    for record in data:
        city = record["city"].replace("'", "''")
        speciality = record["speciality"].replace("'", "''")
        provider_network = record["provider_network"].replace("'", "''")
        cost = record["cost"] or "NULL"
        working_hours = record["working_hours"] or "NULL"

        cursor.execute(
            f"""
            INSERT INTO "services_costs" ("city_id", "speciality_id", "provider_network_id", "cost")
            VALUES (
                (SELECT "id" FROM "cities" WHERE "name" = '{city}'),
                (SELECT "id" FROM "specialities" WHERE "name" = '{speciality}'),
                (SELECT "id" FROM "provider_networks" WHERE "name" = '{provider_network}'),
                {cost}
            )
            """
        )

        cursor.execute(
            f"""
            INSERT INTO "services_working_hours" ("city_id", "speciality_id", "working_hours")
            VALUES (
                (SELECT "id" FROM "cities" WHERE "name" = '{city}'),
                (SELECT "id" FROM "specialities" WHERE "name" = '{speciality}'),
                {working_hours}
            )
            """
        )

    services_counts_data = (
        df.groupby(["city", "speciality", "provider_network"])
        .size()
        .reset_index(name="count")
        .to_dict("records")
    )

    for record in services_counts_data:
        speciality = record["speciality"].replace("'", "''")
        city = record["city"].replace("'", "''")
        provider_network = record["provider_network"].replace("'", "''")
        count = record["count"]

        cursor.execute(
            f"""
            INSERT INTO "services_counts" ("city_id", "speciality_id", "provider_network_id", "count")
            VALUES (
                (SELECT "id" FROM "cities" WHERE "name" = '{city}'),
                (SELECT "id" FROM "specialities" WHERE "name" = '{speciality}'),
                (SELECT "id" FROM "provider_networks" WHERE "name" = '{provider_network}'),
                {count}
            )
            """
        )

    psql_conn.commit()
    cursor.close()
    psql_conn.close()


# TODO: find out a way to make dags don't have a start date, and can be triggered manually
@dag("etl_dag", start_date=datetime.datetime(2000, 1, 1))
def generate_etl_dag():
    vendor1_csv_path = os.path.join(
        os.path.dirname(__file__), "../data/vendor1-data.csv"
    )
    extract_vendor1_csv_data = extract_csv_data_task(
        params={"csv_path": vendor1_csv_path},
    )
    transform_vendor1_csv_data = transform_vendor1_data_task()

    vendor2_csv_path = os.path.join(
        os.path.dirname(__file__), "../data/vendor2-data.csv"
    )
    extract_vendor2_csv_data = extract_csv_data_task(
        params={"csv_path": vendor2_csv_path},
    )
    transform_vendor2_csv_data = transform_vendor2_data_task()

    extract_oltp_data = PostgresOperator(
        task_id="extract_oltp_data_task",
        postgres_conn_id="postgres_oltp",
        sql="""
            SELECT 
                p."FirstName",
                p."LastName",
                s."Name" AS Speciality,
                c."Name" AS City,
                pn."Name" as ProviderNetwork,
                pa_w_hours."AttributeValue" AS W_Hours,
                pa_cost."AttributeValue" AS Cost
            FROM
                "Provider" p
                JOIN "ProviderNetwork" pn on p."ProviderNetworkId" = pn."Id"
                JOIN "ProviderSpecialityAssoc" psa ON p."Id" = psa."ProviderId"
                JOIN "Speciality" s ON psa."SpecialityId" = s."Id"
                JOIN "ProviderAddress" pa ON p."Id" = pa."ProviderId"
                JOIN "City" c ON pa."CityId" = c."Id"
                JOIN "ProviderAttributeValue" pa_w_hours ON (p."Id" = pa_w_hours."ProviderId" AND pa_w_hours."ProviderAttributeKeyId" = 1)
                JOIN "ProviderAttributeValue" pa_cost ON (p."Id" = pa_cost."ProviderId" AND pa_cost."ProviderAttributeKeyId" = 2);
        """,
    )
    transform_oltp_data = transform_oltp_data_task()

    extract_freebase_data = extract_freetext_data_task()

    join_data = join_data_task()

    load_task = load_data_task()

    extract_vendor1_csv_data >> transform_vendor1_csv_data >> join_data >> load_task
    extract_vendor2_csv_data >> transform_vendor2_csv_data >> join_data >> load_task
    extract_oltp_data >> transform_oltp_data >> join_data >> load_task
    extract_freebase_data >> join_data >> load_task


etl_dag = generate_etl_dag()
