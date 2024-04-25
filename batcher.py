import os
import pandas as pd
import time
from io import StringIO
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

def process_csv(file_path: str, batch_size: int, instruction: str, out_csv_file_path: str = None, delay_seconds: int = 0.1, llm: str = 'gpt-3.5-turbo'):
    """
    This function reads a CSV file, batches the rows based on the specified batch size, 
    and prepares the data to be sent to an LLM. It throttles the sending of requests
    by waiting for a specified number of seconds between each batch.

    Parameters:
    file_path (str): The path to the CSV file.
    batch_size (int): The number of rows per batch to be sent in each prompt.
    delay_seconds (int): The number of seconds to wait between sending each batch.

    Returns:
    A Dataframe with the results. If passed a CSV file param, a CSV will also be created
    """

    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Initialize LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", api_key=os.environ.get("OPENAI_API_KEY"))
    
    # Calculate the number of batches
    total_rows = len(df)
    num_batches = (total_rows + batch_size - 1) // batch_size
    print(f"Processing {total_rows} rows in {num_batches} batches...")
    
    # initializing variables
    dataframes = []
    cost = {
        "total_input_tokens": 0,
        "total_output_tokens": 0
    }
    timings = []

    # Process each batch
    for i in range(num_batches):
        start_time = time.time()
        # Extract the batch
        batch_start = i * batch_size
        batch_end = batch_start + batch_size
        batch_df = df.iloc[batch_start:batch_end]
        
        
        # Placeholder: Replace the following line with the actual sending logic
        print(f"Sending batch {i+1}/{num_batches}")
        prompt = generate_prompt(batch_df,instruction)
        result, token_cost = call_LLM_enrich(prompt, llm)

        # TODO append to results
        dataframes.append(result)

        cost["total_input_tokens"] += token_cost["prompt_tokens"]
        cost["total_output_tokens"] += token_cost["completion_tokens"]

        # Wait for a specified delay to throttle the request rate
        time.sleep(delay_seconds)
        end_time = time.time()
        execution_time = end_time - start_time
        timings.append(execution_time)

        if(i % 5 == 0):
            avg_time_per_cycle = sum(timings) / len(timings) / 60
            remaining_cycles = num_batches - i
            time_remaining = remaining_cycles * avg_time_per_cycle
            print(f"Estimated remaining time to process is ~{time_remaining:.1f} mins")

    final_df = pd.concat(dataframes, ignore_index=True)

    print("Finished processing data")
    print("Resulting data information:")
    print(final_df.info())
    print("Total cost:")
    print(cost)

    if out_csv_file_path != None:
        print(f"Generating CSV file on: {out_csv_file_path}")
        final_df.to_csv(out_csv_file_path, index=False)

    return final_df

def generate_prompt(data: pd.DataFrame, instruction: str):
    data_to_send = data.to_json(orient='records')
    prompt = f"{instruction}. Json data:```\n{data_to_send}```"

    # print(f"Prompt:\n{prompt}")
    return prompt

def call_LLM_enrich(prompt: str, llm):
    messages = [
        SystemMessage(content="You will process data provided to you in JSON format, and only return results back in JSON format. Follow the data processing intructions provided by the user. Do not include anything else besides the JSON result"),
        HumanMessage(content=prompt),
    ]
    response = llm.invoke(messages)

    # print(response)

    try:
        json_block_til_end = response.content.split("```json")[1]
        json_block = json_block_til_end.rsplit("```", 1)[0]
        json_data = StringIO(json_block)
        df = pd.read_json(json_data)
        return df, response.response_metadata['token_usage']
    except IndexError:
        raise ValueError(f"Could not find a code block in the response: '{response}'")
    except ValueError as e:
        print(f"Malformed JSON or incompatible JSON format: {e}")

