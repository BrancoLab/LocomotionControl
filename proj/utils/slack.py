from slack import WebClient
from slack.errors import SlackApiError

from proj.secrets import SLACK_TOKEN, SLACK_USER_ID


def send_slack_message(message):
    client = WebClient(token=SLACK_TOKEN)

    try:
        client.chat_postMessage(channel=SLACK_USER_ID, text=message)
    except SlackApiError as e:
        print(f"Got an error: {e.response['error']}")
