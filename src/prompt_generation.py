def make_question_prompt(question: str, asp_facts: str, candidates: str, events: str, TG: list) -> str:
    """
    Build the question prompt given:
    - a question
    - a set of facts derived through an ASP program
    - a set of answer candidates
    - the raw event declarations
    """

    with open('src/prompts/question.txt', 'r') as f:
        question_template = f.read()
    
    # Replace placeholders
    question_prompt = question_template.replace('$QUESTION', question)
    question_prompt = question_prompt.replace('$CANDIDATES', candidates)
    question_prompt = question_prompt.replace('$ASP_FACTS', str(asp_facts))
    question_prompt = question_prompt.replace('$EVENTS', str(events))
    question_prompt = question_prompt.replace('$TG', TG)

    return question_prompt


def query_asp_output_prompt(question: str) -> str:
    """
    Insert question into a prompt asking LLM to query the output of an ASP program
    for relevant output (predicate type choice).
    """

    with open('src/prompts/query_asp_output.txt') as f:
        query_template = f.read()
    
    return query_template.replace('$QUESTION', question)