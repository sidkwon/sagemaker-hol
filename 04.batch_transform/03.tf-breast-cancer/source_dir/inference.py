import json

def input_handler(data, context):
    """ Pre-process request input before it is sent to TensorFlow Serving REST API
    Args:
        data (obj): the request data, in format of dict or string
        context (Context): an object containing request and configuration details
    Returns:
        (dict): a JSON-serializable dict that contains request body and headers
    """
    if context.request_content_type == 'text/csv':
        request = data.read().decode('utf-8').rstrip('\n')
        request = [float(x) for x in request.split(',')]
        request.pop(0) # Remove "id" column
        request.pop(1) # Remove "diagnosis" column
        
        return json.dumps({
            'instances': [request]
        })

    raise ValueError('{{"error": "unsupported content type {}"}}'.format(
        context.request_content_type or "unknown"))


def output_handler(data, context):
    """Post-process TensorFlow Serving output before it is returned to the client.
    Args:
        data (obj): the TensorFlow serving response
        context (Context): an object containing request and configuration details
    Returns:
        (bytes, string): data to return to client, response content type
    """
    if data.status_code != 200:
        raise ValueError(data.content.decode('utf-8'))

    response_content_type = context.accept_header
    
    prediction = json.loads(data.content.decode("utf-8"))['predictions'][0][0]
    output = json.dumps({'predictions': prediction})

    return output, response_content_type
