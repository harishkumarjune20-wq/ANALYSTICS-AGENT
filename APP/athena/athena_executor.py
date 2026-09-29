import boto3
import pandas as pd
import time

from config import (
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    AWS_SESSION_TOKEN,
    AWS_REGION,
    ATHENA_DATABASE,
    ATHENA_OUTPUT_LOCATION
)


athena = boto3.client(
    "athena",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    aws_session_token=AWS_SESSION_TOKEN,
    region_name=AWS_REGION
)


# =====================================================
# AWS SESSION ERROR CHECK
# =====================================================

def is_session_expired_error(error):

    error_text = str(error).lower()

    expired_patterns = [
        "expiredtokenexception",
        "expired token",
        "token is expired",
        "token has expired",
        "security token",
        "expired credentials",
        "invalid security token"
    ]

    return any(
        pattern in error_text
        for pattern in expired_patterns
    )


# =====================================================
# EXECUTE ATHENA QUERY
# =====================================================

def execute_query(sql):

    print("\nExecuting SQL:")
    print(sql)

    # =================================================
    # START QUERY
    # =================================================

    try:

        response = athena.start_query_execution(
            QueryString=sql,
            QueryExecutionContext={
                "Database": ATHENA_DATABASE
            },
            ResultConfiguration={
                "OutputLocation": ATHENA_OUTPUT_LOCATION
            }
        )

    except Exception as e:

        print("\nATHENA START ERROR:")
        if is_session_expired_error(e):
            print("ExpiredTokenException", flush=True)
        else:
            print(str(e), flush=True)

        if is_session_expired_error(e):

            return {
                "success": False,
                "error_type": "SESSION_EXPIRED",
                "message": (
                    "AWS session token has expired. "
                    "Please refresh your AWS credentials."
                )
            }

        return {
            "success": False,
            "error_type": "ATHENA_ERROR",
            "message": (
                "Unable to start the Athena query."
            )
        }

    query_id = response["QueryExecutionId"]

    print(f"\nQuery ID: {query_id}")

    # =================================================
    # WAIT FOR QUERY
    # =================================================

    max_wait_time = 120
    elapsed_time = 0
    poll_interval = 2

    while elapsed_time < max_wait_time:

        try:

            result = athena.get_query_execution(
                QueryExecutionId=query_id
            )

        except Exception as e:

            print("\nATHENA STATUS ERROR:")
            if is_session_expired_error(e):
                print("ExpiredTokenException", flush=True)
            else:
                print(str(e), flush=True)

            if is_session_expired_error(e):

                return {
                    "success": False,
                    "error_type": "SESSION_EXPIRED",
                    "message": (
                        "🔐 AWS session token has expired. "
                        "Please refresh your AWS credentials and try again."
                    )
                }

            return {
                "success": False,
                "error_type": "ATHENA_ERROR",
                "message": (
                    "Unable to check the Athena query status."
                )
            }

        status = result["QueryExecution"]["Status"]

        state = status["State"]

        print(
            f"Athena Query Status: {state}"
        )

        # ---------------------------------------------
        # SUCCESS
        # ---------------------------------------------

        if state == "SUCCEEDED":

            print("Query succeeded.")

            break

        # ---------------------------------------------
        # ATHENA QUERY FAILURE
        # ---------------------------------------------

        if state in [
            "FAILED",
            "CANCELLED"
        ]:

            error_message = status.get(
                "StateChangeReason",
                "Unknown Athena Error"
            )

            return {
                "success": False,
                "error_type": "ATHENA_QUERY_ERROR",
                "message": error_message
            }

        time.sleep(
            poll_interval
        )

        elapsed_time += poll_interval

    else:

        return {
            "success": False,
            "error_type": "ATHENA_TIMEOUT",
            "message": (
                f"Athena query timed out after "
                f"{max_wait_time} seconds."
            )
        }

    # =================================================
    # GET QUERY RESULTS
    # =================================================

    try:

        results = athena.get_query_results(
            QueryExecutionId=query_id
        )

    except Exception as e:

        print("\nATHENA RESULT ERROR:")
        if is_session_expired_error(e):
            print("ExpiredTokenException", flush=True)
        else:
            print(str(e), flush=True)

        if is_session_expired_error(e):

            return {
                "success": False,
                "error_type": "SESSION_EXPIRED",
                "message": (
                    "AWS session token has expired. "
                    "Please refresh your AWS credentials."
                )
            }

        return {
            "success": False,
            "error_type": "ATHENA_ERROR",
            "message": (
                "Unable to retrieve Athena query results."
            )
        }

    # =================================================
    # CONVERT RESULT TO DATAFRAME
    # =================================================

    rows = results["ResultSet"]["Rows"]

    if len(rows) == 0:

        return pd.DataFrame()

    columns = [
        col.get(
            "VarCharValue",
            ""
        )
        for col in rows[0]["Data"]
    ]

    data = []

    for row in rows[1:]:

        values = []

        for cell in row["Data"]:

            values.append(
                cell.get(
                    "VarCharValue",
                    ""
                )
            )

        data.append(values)

    df = pd.DataFrame(
        data,
        columns=columns
    )

    return df