from utils import *

def my_generate_prompt_TG_trans(dataset_name, story, TG, entities, relation, times, f_ICL, f_shorten_story, f_hard_mode, 
                                transferred_dataset_name, mode=None, eos_token="</s>", max_story_len=1500, prompt_format='plain'):
    '''
    Generate the prompt for text to TG translation (given context and keywords, generate the relevant TG)

    Args:
    - story: str or list, the story
    - TG: str or list, the TG
    - entities: str or list, the entities
    - relation: str, the relation
    - times: str or list, the times
    - f_ICL: bool, whether to use ICL
    - f_shorten_story: bool, whether to shorten the story
    - f_hard_mode: bool, whether to use hard mode
    - transferred_dataset_name: str, the name of the transferred dataset
    - mode: train or test
    - eos_token: str, the end of sentence token
    - max_story_len: int, the maximum length of the story (only valid when f_shorten_story is True)
    - prompt_format: str, the format of the prompt

    Returns:
    - prompt: str, the prompt
    '''
    assert prompt_format.lower() in ['plain', 'json'], "Prompt format is not recognized."

    def add_examples_in_prompt(prompt, prompt_format):
        if f_ICL and mode == 'test':
            filename_suffix = '_json' if prompt_format.lower() == 'json' else ''
            file_path = f'../materials/{dataset_name}/prompt_examples_text_to_TG_Trans{filename_suffix}.txt' if (not f_hard_mode) else \
                        f'../materials/{dataset_name}/prompt_examples_text_to_TG_Trans_hard{filename_suffix}.txt'
            with open(file_path) as txt_file:
                prompt_examples = txt_file.read()
            prompt = f"\n\n{prompt_examples}\n\nTest:\n{prompt}"
        return prompt.strip() + '\n'


    # Convert the list to string
    entities = ' , '.join(add_brackets(entities)) if entities is not None else None
    times = ' , '.join(add_brackets(times)) if times is not None else None

    if f_shorten_story:
        story = shorten_story(story, max_story_len)
 
    if relation is None:
        # If we do not have such information extracted from the questions, we will translate the whole story.
        prompt = add_examples_in_prompt(f'### Input:\n{{\n"Story": {json.dumps(story)},\n"Instruction": "Summary all the events as a timeline. Only return me json."\n}}\n ### Output: \n```json', prompt_format) if prompt_format.lower() == 'json' else \
                add_examples_in_prompt(f"### Input:\n{story}\n\nSummary all the events as a timeline.\n\n ### Output:", prompt_format)
    else:
        if f_hard_mode or entities is None or times is None:
            prompt = add_examples_in_prompt(f'### Input:\n{{\n"Story": {json.dumps(story)},\n"Instruction": "Summary {relation} as a timeline. Only return me json."\n}}\n ### Output: \n```json', prompt_format) if prompt_format.lower() == 'json' else \
                    add_examples_in_prompt(f"### Input:\n{story}\n\nSummary {relation} as a timeline.\n\n ### Output:", prompt_format)
        else:
            prompt = add_examples_in_prompt(f'### Input:\n{{\n"Story": {json.dumps(story)},\n"Instruction": "Given the time periods: {times}, summary {relation} as a timeline. Choose from {entities}. Only return me json."\n}}\n ### Output: \n```json', prompt_format) if prompt_format.lower() == 'json' else \
                    add_examples_in_prompt(f"### Input:\n{story}\n\nGiven the time periods: {times}, summary {relation} as a timeline. Choose from {entities}.\n\n ### Output:", prompt_format)
             
    # For training data, we provide the TG as label.
    if TG is not None:
        # If we want to test the transfer learning performance, we can change the format of the TG in TGQA to other datasets.
        TG = TG_formating_change(TG, dataset_name, transferred_dataset_name)
        timeline = "\n".join(TG)
        prompt += f'{{\n"Timeline":\n{json.dumps(TG)}\n}}\n```' if prompt_format.lower() == 'json' else f"Timeline:\n{timeline}\n"

    prompt += eos_token
    return prompt