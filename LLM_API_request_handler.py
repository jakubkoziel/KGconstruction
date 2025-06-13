from openai import OpenAI
import time


class RequestHandler:
    def __init__(self, api_key, **kwargs):
        self.client = OpenAI(api_key=api_key, **kwargs)

    def send_request(self, model, messages, process_id):
        retries = 0
        max_retires = 7

        while retries <= max_retires:
            try:
                if model != 'o1-mini':
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=0.0,
                        # response_format={"type": "json_object"}  # enforce JSON output
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        # temperature=0.0,
                        # response_format={"type": "json_object"}  # enforce JSON output
                    )

                return response.choices[0].message.content

            except Exception as e:
                print(f'Process {process_id} failed on sending request. Retry={retries} with error: {e}')
                if retries <= max_retires - 1:
                    time.sleep(2 ** retries)
                else:
                    raise Exception('Too much processes')
                retries += 1

        raise Exception('Too much processes')
