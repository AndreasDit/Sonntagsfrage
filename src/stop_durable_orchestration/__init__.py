import azure.functions as func
import azure.durable_functions as df

async def main(req: func.HttpRequest, starter: str) -> func.HttpResponse:
    client = df.DurableOrchestrationClient(starter)

    instance_id = 'c68afb46de874dd496abff857cdbfc44'
    reason = "It was time to be done."
    return client.terminate(instance_id, reason)