"""Prompts of the Turkish evaluation: translation, system prompts for the decoder, and the LLM judge."""

from memorilla.benchmarks import PERSONA_PROMPT

PERSONA_PROMPT_TR = (
    "Kişiselleştirilmiş bir yapay zekâ asistanısın. Kullanıcıyla ilgili soruyu, kullanıcı hakkındaki bilgine dayanarak "
    "yanıtla."
)
REPLY_IN_TURKISH = " Türkçe yanıt ver."
PERSONA_PROMPT_REPLY_TR = PERSONA_PROMPT + REPLY_IN_TURKISH

DOCUMENT_TEMPLATE = """Translate the text below from English into Turkish.

The text is part of a conversation between a user and an AI assistant. Write natural, fluent Turkish as a native speaker would, keeping the tone and register of each speaker. Keep speaker labels at the start of a turn (such as "Leo:" or "Assistant:") and personal names exactly as they are, keep line breaks and formatting, and translate everything else. Do not add, remove, summarise or explain anything.

Output only the translation.

Text:
{text}"""

QA_TEMPLATE = """Translate a question and its {num_options} answer options from English into Turkish.

Write natural, fluent Turkish as a native speaker would. Keep the point of view of the source (for example, whether it addresses the user as "you" or describes them in the third person) and keep personal names unchanged. Translate every option with the same care, register and level of detail, and keep their lengths and structure comparable, so that the translation does not make any option stand out. Options that differ in English must stay different in Turkish.

Return only a JSON object of the form {{"question": "...", "options": ["...", ...]}} with exactly {num_options} options in their original order.

Input:
{payload}"""

JUDGE_TEMPLATE = """You are evaluating a personalised AI assistant. The assistant has a memory of its past conversations with a user and has just answered a question about that user. A reference answer, written by someone who knows the user, shows what the assistant should have inferred.

Rate the assistant's answer on each criterion with an integer from 1 (very poor) to 5 (excellent):
- "personalization": does the answer rely on the specific preferences, circumstances or habits of the user that the reference relies on, rather than giving generic advice?
- "consistency": does the answer agree with the reference? It may be worded differently or add compatible details.
- "no_fabrication": does the answer avoid inventing facts about the user that the reference does not support? (5 = nothing invented)
- "grammar": is the answer grammatical in the language of the question? For Turkish, check suffixes, vowel harmony, case endings and agreement.
- "naturalness": does it read like text a native speaker of that language would write, without translationese or words from another language?
- "address": does it address the user in one consistent, appropriate way? For Turkish, it should use either "sen" or "siz", not a mix.

Also report "language": the main language of the answer as an ISO 639-1 code.

Return only a JSON object: {{"personalization": n, "consistency": n, "no_fabrication": n, "grammar": n, "naturalness": n, "address": n, "language": "xx"}}

Question:
{question}

Reference answer:
{reference}

Assistant's answer:
{prediction}"""


def document_prompt(text: str) -> str:
    """Fill the document-translation prompt.

    Args:
        text: One document (a conversation or a chunk of one).

    Returns:
        The prompt.
    """
    return DOCUMENT_TEMPLATE.format(text=text)


def qa_prompt(payload: str, num_options: int) -> str:
    """Fill the question-and-options translation prompt.

    Args:
        payload: JSON object with the question and its options.
        num_options: Number of options in the payload.

    Returns:
        The prompt.
    """
    return QA_TEMPLATE.format(payload=payload, num_options=num_options)


def judge_prompt(question: str, reference: str, prediction: str) -> str:
    """Fill the judge prompt.

    Args:
        question: The question the assistant answered.
        reference: The reference answer.
        prediction: The assistant's answer.

    Returns:
        The prompt.
    """
    return JUDGE_TEMPLATE.format(question=question, reference=reference, prediction=prediction)
