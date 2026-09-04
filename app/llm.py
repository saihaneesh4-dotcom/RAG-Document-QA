import os

from dotenv import load_dotenv
from google import genai
from google.genai import errors


load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


def ask_llm(prompt):

    try:

        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=prompt
        )

        return response.text

    except errors.ClientError as error:

        if error.code == 429:

            return (
                "⚠️ Gemini API quota has been reached. "
                "Please try again later."
            )

        if error.code == 404:

            return (
                "⚠️ The configured Gemini model is "
                "currently unavailable."
            )

        return (
            "⚠️ Gemini could not generate an answer "
            "at this time."
        )

    except Exception:

        return (
            "⚠️ An unexpected error occurred while "
            "generating the answer."
        )