# This function an HTTP starter function for Durable Functions.
# Before running this sample, please:
# - create a Durable orchestration function
# - create a Durable activity function (default name is "Hello")
# - add azure-functions-durable to requirements.txt
# - run pip install -r requirements.txt

import logging
import json
import azure.functions as func
import azure.durable_functions as df


async def main(req: func.HttpRequest, starter: str) -> func.HttpResponse:
    client = df.DurableOrchestrationClient(starter)

    instances = await client.get_status_all()

    for instance in instances:
        if instance.to_json()['runtimeStatus'] == 'Running':
            instance_id = instance.to_json()['instanceId']
            reason = 'it was time'
            # print(instance)
            logging.info(instance)
            # logging.log(1, json.dumps(instance))
            logging.info(json.dumps(instance.to_json()))
            print(json.dumps(instance.to_json()))
            print(instance_id)
            await client.terminate(instance_id, reason)
            print(client.terminate(instance_id, reason))
            logging.info(client.terminate(instance_id, reason))
    # return instances