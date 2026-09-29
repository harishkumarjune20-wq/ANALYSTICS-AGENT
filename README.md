# ANALYSTICS-AGENT
# AI-Powered Analytics Agent

An interactive analytics assistant that allows users to ask questions about data in natural language. The application uses a Streamlit chat interface, a LangGraph workflow, Amazon Athena for SQL execution, and language models to generate SQL and explain query results.

## Features

* **Natural-Language Analytics:** Ask questions about data using everyday language.
* **Intent Routing:** Route requests to analytics, conversation, business-analysis, or visualization paths.
* **Text-to-SQL:** Generate Athena-compatible SQL from analytics questions.
* **SQL Validation and Repair:** Validate generated SQL and attempt to repair invalid queries.
* **Amazon Athena Execution:** Execute SQL queries and retrieve results from Athena.
* **Data Visualization:** Display query results in tables and visualizations.
* **AI-Generated Insights:** Summarize results and highlight patterns in the returned data.
* **Follow-Up Questions:** Maintain conversation and analytics context to support follow-up requests.
* **Conversation Support:** Respond to greetings and general messages without executing data queries.
* **Error Handling:** Handle query failures and model errors through recovery paths.
* **Interactive Interface:** Review conversations, query results, and analysis in one place.

## How It Works

1. The user enters a question through the Streamlit chat interface.
2. The LangGraph workflow interprets the request and determines the appropriate route.
3. For data questions, the SQL-generation component creates an Athena query.
4. The query passes through SQL validation.
5. If validation fails, the repair step attempts to correct the SQL.
6. The validated query is executed in Amazon Athena.
7. The returned data is processed and displayed.
8. The agent generates relevant summaries, insights, or visualizations.

Conversation and other supported requests can follow separate workflow paths without executing a data query.

## Technology Stack

* Python
* Streamlit
* LangGraph
* Amazon Athena
* Boto3
* Pandas and NumPy
* Plotly
* SQLGlot
* Ollama for local model inference
* NVIDIA Nemotron API for model-powered analysis
* Requests and urllib3
* Python-dotenv

## Project Structure

```text
cs/
├── agents/
│   ├── analytics_insights.py
│   ├── conversation_agent.py
│   ├── error_recovery.py
│   ├── intent_classifier.py
│   ├── response_generator.py
│   ├── sql_generator_athena.py
│   ├── suggestion_generator.py
│   └── syntax_checker.py
├── athena/
│   └── athena_executor.py
├── graph/
│   ├── adapter.py
│   ├── analytics_graph.py
│   ├── nodes.py
│   ├── routers.py
│   └── state.py
├── utils/
│   ├── nemotron_client.py
│   ├── ollama_client.py
│   └── schema_loader_athena.py
├── assets/
│   ├──YOUR_LOGO.PNG 
│   ├──YOUR_LOGO.PNG
│   └──SIDE BAR BACK GROUND.PNG
├── app_1.py
├── config.py
├── requirements.txt
└── schema_cache.txt
```

## Prerequisites

Before running the application, ensure you have:

* Python 3.10 or a compatible version.
* An AWS account with access to Amazon Athena.
* An S3 location for Athena query results.
* AWS credentials with the required permissions.
* Ollama and a locally installed model for local inference.
* An NVIDIA API key for the Nemotron integration.

## Installation

### 1. Extract the Project

Extract the project files and open a terminal in the directory containing `app_1.py` and `requirements.txt`.

### 2. Create a Virtual Environment

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Configuration

### AWS and Athena Configuration

Create a `.env` file in the project root.

Add the following configuration:

```env
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_SESSION_TOKEN=
AWS_REGION=your_aws_region
ATHENA_DATABASE=your_athena_database
ATHENA_OUTPUT_LOCATION=s3://your-bucket/athena-results/
```

Use an IAM role or AWS profile where possible.

If you use temporary AWS credentials, provide the current session token.

Never commit credentials or secrets to version control.

### Data Schema Configuration

The schema loader reads `schema_cache.txt`.

Update this file with the table and column definitions for your Athena data source.

Review `utils/schema_loader_athena.py` to ensure the table reference matches your intended database and table.

### Ollama Configuration

The Ollama client uses the local API endpoint:

```text
http://127.0.0.1:11434/api/generate
```

The model name is configured in:

```text
utils/ollama_client.py
```

Install and start Ollama, then ensure the configured model is available.

### NVIDIA Nemotron Configuration

Add your NVIDIA API key to the `.env` file:

```env
NVIDIA_API_KEY=your_nvidia_api_key
```

The model configuration is available in:

```text
utils/nemotron_client.py
```

Keep your API key private.

## Run the Application

From the project root, execute:

```bash
streamlit run app_1.py
```

Open the local URL displayed in the terminal to access the analytics assistant.

## Example Questions

The following questions demonstrate the types of analytics supported, depending on your data schema:

* Show the total sales by month.
* Count records by status.
* Show the top 10 categories by total amount.
* Compare results across months.
* Break down the results by region.
* Create a chart showing the trend over time.
* Summarize the main patterns in the query results.
* Filter the previous result to show only a specific status.

## Troubleshooting

### AWS Authentication Errors

* Check that your AWS credentials are valid.
* Verify that temporary credentials have not expired.
* Confirm the configured AWS region and permissions.

### Athena Query Errors

* Check table and column names.
* Verify the Athena database and S3 output location.
* Ensure the generated SQL is compatible with Athena.

### Ollama Connection Errors

* Ensure Ollama is running locally.
* Verify that the configured model is installed.
* Check the API endpoint.

### NVIDIA API Errors

* Verify that `NVIDIA_API_KEY` is configured correctly.
* Check the model name and API access.

### Incorrect SQL or Results

* Ensure the schema cache matches the actual data source.
* Review the generated SQL and confirm that filters and fields match the question.
* Validate the output against the underlying data.

## Security and Data Handling

* Never commit `.env` files, AWS credentials, API keys, or session tokens.
* Use least-privilege AWS permissions.
* Avoid exposing sensitive data in logs or screenshots.
* Review generated SQL and AI-generated summaries before relying on them.
* Follow the data access and retention requirements applicable to your environment.

## Limitations

* SQL quality depends on the accuracy of the schema and the language model.
* Results depend on Athena permissions and available data.
* AI-generated insights should be verified against the underlying query results.
* Model response times and availability vary by provider.
* Environment-specific configuration may be required before running the application against a different dataset.

## License

Add the license and usage terms applicable to this project before distributing it.
