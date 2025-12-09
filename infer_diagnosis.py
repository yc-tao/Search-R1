import pickle
import requests
import re
import os
from pathlib import Path
from openai import OpenAI

# Configuration
DATA_PATH = os.path.expanduser("~/SRL/data/concatenated_notes_by_episode.pkl")
RETRIEVAL_SERVER_URL = "http://127.0.0.1:56321/retrieve"
VLLM_SERVER_URL = "http://127.0.0.1:60363/v1"  # vLLM OpenAI-compatible API endpoint
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
TOPK_RETRIEVAL = 3
USE_VLLM = True  # Set to True to use vLLM server, False for local model

# Phenotype list for classification
PHENOTYPES = [
    'Acute and unspecified renal failure',
    'Acute cerebrovascular disease',
    'Acute myocardial infarction',
    'Cardiac dysrhythmias',
    'Chronic kidney disease',
    'Chronic obstructive pulmonary disease and bronchiectasis',
    'Complications of surgical procedures or medical care',
    'Conduction disorders',
    'Congestive heart failure; nonhypertensive',
    'Coronary atherosclerosis and other heart disease',
    'Diabetes mellitus with complications',
    'Diabetes mellitus without complication',
    'Disorders of lipid metabolism',
    'Essential hypertension',
    'Fluid and electrolyte disorders',
    'Gastrointestinal hemorrhage',
    'Hypertension with complications and secondary hypertension',
    'Other liver diseases',
    'Other lower respiratory disease',
    'Other upper respiratory disease',
    'Pleurisy; pneumothorax; pulmonary collapse',
    'Pneumonia (except that caused by tuberculosis or sexually transmitted disease)',
    'Respiratory failure; insufficiency; arrest (adult)',
    'Septicemia (except in labor)',
    'Shock'
]

# vLLM client setup
if USE_VLLM:
    vllm_client = OpenAI(
        api_key="EMPTY",  # vLLM doesn't require API key
        base_url=VLLM_SERVER_URL,
    )

curr_search_template = '\n\n{output_text}<information>{search_results}</information>\n\n'

# Import torch and transformers only if not using vLLM
if not USE_VLLM:
    import torch
    import transformers
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    curr_eos = [151645, 151643]  # for Qwen2.5 series models

    # Define the custom stopping criterion
    class StopOnSequence(transformers.StoppingCriteria):
        def __init__(self, target_sequences, tokenizer):
            # Encode the string so we have the exact token-IDs pattern
            self.target_ids = [tokenizer.encode(target_sequence, add_special_tokens=False) for target_sequence in target_sequences]
            self.target_lengths = [len(target_id) for target_id in self.target_ids]
            self._tokenizer = tokenizer

        def __call__(self, input_ids, scores, **kwargs):
            # Make sure the target IDs are on the same device
            targets = [torch.as_tensor(target_id, device=input_ids.device) for target_id in self.target_ids]

            if input_ids.shape[1] < min(self.target_lengths):
                return False

            # Compare the tail of input_ids with our target_ids
            for i, target in enumerate(targets):
                if torch.equal(input_ids[0, -self.target_lengths[i]:], target):
                    return True

            return False

def load_data(data_path):
    """Load the concatenated notes from pickle file."""
    print(f"Loading data from {data_path}...")

    # Compatibility shim for NumPy 2.x pickles with NumPy 1.x
    import sys
    import numpy as np
    sys.modules['numpy._core'] = np.core
    sys.modules['numpy._core.numeric'] = np.core.numeric

    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    print(f"Loaded {len(data)} episodes")
    return data

def extract_last_section(concatenated_note):
    """
    Extract the last section of the note split by '=' separator lines.

    Args:
        concatenated_note: The full concatenated note text

    Returns:
        The last section of the note, or empty string if no separator found
    """
    separator = "=" * 80  # Looking for lines of 80 equal signs
    sections = concatenated_note.split(separator)

    # Filter out empty sections
    sections = [s.strip() for s in sections if s.strip()]

    if sections:
        return sections[-1]
    else:
        return ""

def split_and_format_documents(concatenated_note, episode_id):
    """
    Split concatenated note into chunks with the following rules:
    1. Each chunk must be at least 50 words in length
    2. Splits can only occur at punctuation marks
    3. Each chunk is extended to include one sentence before and after for context

    Args:
        concatenated_note: The full concatenated note text
        episode_id: The episode identifier

    Returns:
        List of document dictionaries formatted for retrieval server
    """
    # Split text into sentences at punctuation marks
    # Using regex to split at . ! ? followed by space or newline
    sentence_pattern = re.compile(r'([.!?]+[\s\n]+)')
    parts = sentence_pattern.split(concatenated_note)

    # Reconstruct sentences (combining text with their punctuation)
    sentences = []
    for i in range(0, len(parts) - 1, 2):
        if i + 1 < len(parts):
            sentence = parts[i] + parts[i + 1]
            sentences.append(sentence.strip())
    # Add last part if it doesn't end with punctuation
    if len(parts) % 2 == 1 and parts[-1].strip():
        sentences.append(parts[-1].strip())

    # Filter out empty sentences
    sentences = [s for s in sentences if s]

    if not sentences:
        print(f"Warning: No sentences found in episode {episode_id}")
        return []

    # Group sentences into chunks of at least 50 words
    MIN_WORDS = 50
    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        word_count = len(sentence.split())
        current_chunk.append(sentence)
        current_word_count += word_count

        # If we have at least 50 words, finalize this chunk
        if current_word_count >= MIN_WORDS:
            chunks.append(current_chunk)
            current_chunk = []
            current_word_count = 0

    # Add remaining sentences as a chunk if any
    if current_chunk:
        chunks.append(current_chunk)

    # Format as documents with context extension (one sentence before and after)
    documents = []
    for idx, chunk in enumerate(chunks):
        # Find the index range of this chunk in the original sentences list
        chunk_start_idx = None
        for i in range(len(sentences)):
            if sentences[i] == chunk[0]:
                chunk_start_idx = i
                break

        if chunk_start_idx is None:
            continue

        chunk_end_idx = chunk_start_idx + len(chunk) - 1

        # Extend with one sentence before and after
        extended_start = max(0, chunk_start_idx - 1)
        extended_end = min(len(sentences) - 1, chunk_end_idx + 1)

        # Build the extended chunk
        extended_sentences = sentences[extended_start:extended_end + 1]
        extended_text = ' '.join(extended_sentences)

        doc = {
            "id": f"episode_{episode_id}_piece_{idx}",
            "contents": f"\"Note Piece {idx}\"\n{extended_text}"
        }
        documents.append(doc)

    print(f"Split episode {episode_id} into {len(documents)} pieces (min {MIN_WORDS} words per chunk)")
    return documents

def get_query(text):
    """Extract query from <search> tags."""
    pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        return matches[-1].strip()
    else:
        return None

def extract_reasoning(text):
    """Extract all reasoning blocks from <think> tags."""
    pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    matches = pattern.findall(text)
    return matches

def extract_answer(text):
    """Extract answer from <answer> tags."""
    pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        return matches[-1].strip()
    else:
        return None

def summarize_last_section_vllm(last_section):
    """
    Summarize the last section of the note, focusing on diagnoses.

    Args:
        last_section: The last section of the concatenated note

    Returns:
        A brief summary covering all diagnoses made in this section
    """
    if not last_section.strip():
        return "No last section found to summarize."

    prompt = f"""Please provide a brief summary of the following medical note. Be concise but comprehensive.

Medical Note Section:
{last_section}

Summary of diagnoses and conditions:"""

    try:
        response = vllm_client.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.0,
        )

        summary = response.choices[0].message.content.strip()
        return summary

    except Exception as e:
        print(f"Error summarizing last section: {e}")
        return "Error generating summary."

def summarize_last_section_local(last_section, tokenizer, model):
    """
    Summarize the last section of the note using local model.

    Args:
        last_section: The last section of the concatenated note
        tokenizer: The tokenizer
        model: The LLM model

    Returns:
        A brief summary covering all diagnoses made in this section
    """
    if not last_section.strip():
        return "No last section found to summarize."

    prompt = f"""Please provide a brief summary of the following medical note section. Focus specifically on listing ALL diagnoses, medical conditions, and health issues mentioned. Be concise but comprehensive.

Medical Note Section:
{last_section}

Summary of diagnoses and conditions:"""

    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False
        )

    try:
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        attention_mask = torch.ones_like(input_ids)

        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=512,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
            temperature=0.0
        )

        generated_tokens = outputs[0][input_ids.shape[1]:]
        summary = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        return summary

    except Exception as e:
        print(f"Error summarizing last section: {e}")
        return "Error generating summary."

def search(query: str, documents: list):
    """
    Search for relevant documents given a query.

    Args:
        query: The search query string
        documents: List of documents to search within

    Returns:
        Tuple of (formatted string of search results, raw results for tracking)
    """
    payload = {
        "queries": [query],
        "documents": documents,
        "topk": TOPK_RETRIEVAL,
        "return_scores": True
    }

    try:
        response = requests.post(RETRIEVAL_SERVER_URL, json=payload)
        response_json = response.json()
        results = response_json['result']
    except Exception as e:
        print(f"Error calling retrieval server: {e}")
        print("Make sure the retrieval server is running with: bash retrieval_launch.sh")
        raise

    def _passages2string(retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            score = doc_item.get('score', 'N/A')
            format_reference += f"Doc {idx+1}(Title: {title}, Score: {score:.2f}) {text}\n"
        return format_reference

    return _passages2string(results[0]), results[0]

def run_inference_vllm(question, documents, initial_summary="", max_searches=50):
    """
    Run iterative inference with retrieval using vLLM server with chat completion mode.

    Args:
        question: The diagnosis question
        documents: List of document pieces to search from
        initial_summary: Summary of the last section of the note (optional)
        max_searches: Maximum number of search iterations

    Returns:
        Dictionary containing full results
    """
    # Prepare the initial user message
    initial_prompt = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can provide the final answer inside <answer> and </answer>. Question: {question} \
"""

    # Add initial summary if available
    if initial_summary:
        initial_prompt += f"""

Initial Summary from Last Section of Note:
{initial_summary}
"""

    initial_prompt += "\n"

    # Initialize messages list with user role (similar to local model's chat template)
    messages = [{"role": "user", "content": initial_prompt}]

    print('\n\n################# [Start Reasoning + Searching] ##################\n\n')
    print(f"Initial user message:\n{initial_prompt}")

    # Track all searches and results
    search_history = []
    retrieved_passages = []
    full_output = ""
    iteration = 0

    # Stop sequences for vLLM
    stop_sequences = ["</search>"]

    while iteration < max_searches:
        try:
            # Call vLLM server with chat completion API
            response = vllm_client.chat.completions.create(
                model=MODEL_ID,
                messages=messages,
                max_tokens=1024,
                temperature=0.0,
                stop=stop_sequences,
                extra_body={
                    "stop_token_ids": [151645, 151643]  # Qwen EOS tokens
                }
            )

            output_text = response.choices[0].message.content

            # Add </search> back if it was removed by stop sequence
            if "<search>" in output_text and "</search>" not in output_text:
                output_text += "</search>"

            # Check if we hit EOS (no </search> found)
            if response.choices[0].finish_reason == "stop" and "</search>" not in output_text:
                full_output += output_text
                print(output_text)
                # Add final assistant message to conversation history
                messages.append({"role": "assistant", "content": output_text})
                break

            full_output += output_text
            print(output_text)

            # Add assistant message to conversation
            messages.append({"role": "assistant", "content": output_text})

            # Extract query from the assistant's output
            tmp_query = get_query(output_text)

            if tmp_query:
                print(f'\n[Search iteration {iteration + 1}] Query: "{tmp_query}"')
                search_results, raw_results = search(tmp_query, documents=documents)
                search_history.append(tmp_query)
                retrieved_passages.extend(raw_results)
            else:
                search_results = ''

            # Format search results as user message (system provides information)
            search_info = f"<information>{search_results}</information>\n"
            messages.append({"role": "user", "content": search_info})
            messages.append({"role": "user", "content": "If no information is found useful, approach the question in a different way, and search with a different query."}) 

            full_output += search_info
            iteration += 1
            print(f"\n{search_info}")

        except Exception as e:
            print(f"Error calling vLLM server: {e}")
            print("Make sure the vLLM server is running")
            raise

    # Extract reasoning and answer
    reasoning_blocks = extract_reasoning(full_output)
    final_answer = extract_answer(full_output)

    if final_answer is None:
        print("\nNo final answer found in <answer> tags.")
        return run_inference_vllm(question, documents, initial_summary=initial_summary, max_searches=max_searches)

    return {
        'full_output': full_output,
        'search_history': search_history,
        'retrieved_passages': retrieved_passages,
        'reasoning_blocks': reasoning_blocks,
        'final_answer': final_answer,
        'num_searches': len(search_history),
        'initial_summary': initial_summary
    }

def run_inference(question, documents, tokenizer, model, stopping_criteria, initial_summary="", max_searches=10):
    """
    Run iterative inference with retrieval.

    Args:
        question: The diagnosis question
        documents: List of document pieces to search from
        tokenizer: The tokenizer
        model: The LLM model
        stopping_criteria: Stopping criteria for generation
        initial_summary: Summary of the last section of the note (optional)
        max_searches: Maximum number of search iterations

    Returns:
        Dictionary containing full results
    """
    # Prepare the prompt
    prompt = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can provide the final answer inside <answer> and </answer>. \
List all medical conditions that occurred during the hospitalization. \
Question: {question}"""

    # Add initial summary if available
#     if initial_summary:
#         prompt += f"""

# Initial Summary from Last Section of Note:
# {initial_summary}
# """

    prompt += "\n"

    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False
        )

    print('\n\n################# [Start Reasoning + Searching] ##################\n\n')
    print(prompt)

    # Track all searches and results
    search_history = []
    retrieved_passages = []
    full_output = ""
    iteration = 0

    while iteration < max_searches:
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        attention_mask = torch.ones_like(input_ids)

        # Generate text with the stopping criteria
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1024,
            stopping_criteria=stopping_criteria,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
            temperature=0.7
        )

        # Check if we hit EOS token
        if outputs[0][-1].item() in curr_eos:
            generated_tokens = outputs[0][input_ids.shape[1]:]
            output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            full_output += output_text
            print(output_text)
            break

        generated_tokens = outputs[0][input_ids.shape[1]:]
        output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Extract query from the full context
        tmp_query = get_query(tokenizer.decode(outputs[0], skip_special_tokens=True))

        if tmp_query:
            print(f'\n[Search iteration {iteration + 1}] Query: "{tmp_query}"')
            search_results, raw_results = search(tmp_query, documents=documents)
            search_history.append(tmp_query)
            retrieved_passages.extend(raw_results)
        else:
            search_results = ''

        search_text = curr_search_template.format(output_text=output_text, search_results=search_results)
        prompt += search_text
        full_output += search_text
        iteration += 1
        print(search_text)

    # Extract reasoning and answer
    reasoning_blocks = extract_reasoning(full_output)
    final_answer = extract_answer(full_output)

    return {
        'full_output': full_output,
        'search_history': search_history,
        'retrieved_passages': retrieved_passages,
        'reasoning_blocks': reasoning_blocks,
        'final_answer': final_answer,
        'num_searches': len(search_history),
        'initial_summary': initial_summary
    }

def classify_phenotypes_vllm(diagnosis_answer):
    """
    Classify which phenotypes the patient has based on the diagnosis answer.
    Uses per-phenotype classification to get logprobs for each phenotype.

    Args:
        diagnosis_answer: The diagnosis text from <answer> tags

    Returns:
        List of dictionaries with phenotype name, prediction (0 or 1), and logprob
    """
    import math
    results = []

    for i, phenotype in enumerate(PHENOTYPES):
        prompt = f"based on the following diagnosis, does the patient have {phenotype}? diagnosis{diagnosis_answer}. answer in 'yes' or 'no' only."
        
        print(f"[Debug] Prompt for phenotype classification: {prompt}")

        try:
            response = vllm_client.chat.completions.create(
                model=MODEL_ID,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5,
                temperature=0.0,
                logprobs=True,
                top_logprobs=20  # Get top 20 to ensure we capture Yes/No tokens
            )

            answer_text = response.choices[0].message.content.strip()

            # print(f"Debug: answer_text='{answer_text}'")
            # import pdb; pdb.set_trace()

            # Get the first token's logprobs
            if response.choices[0].logprobs and response.choices[0].logprobs.content:
                first_token_logprobs = response.choices[0].logprobs.content[0]

                # Find logprobs for "Yes" and "No" tokens
                yes_logprob = None
                no_logprob = None

                for top_logprob in first_token_logprobs.top_logprobs:
                    token = top_logprob.token.strip().lower()
                    if token in ["yes", "yes."]:
                        yes_logprob = top_logprob.logprob
                    elif token in ["no", "no."]:
                        no_logprob = top_logprob.logprob
                    if yes_logprob is not None and no_logprob is not None:
                        break

                # Determine prediction and logprob
                if "yes" in answer_text.lower():
                    prediction = 1
                    logprob = yes_logprob if yes_logprob is not None else first_token_logprobs.logprob
                else:
                    prediction = 0
                    logprob = no_logprob if no_logprob is not None else first_token_logprobs.logprob

                # Convert logprob to probability for display
                probability = math.exp(logprob)

                results.append({
                    'phenotype': phenotype,
                    'prediction': prediction,
                    'logprob': logprob,
                    'probability': probability
                })

                print(f"[{i+1}/{len(PHENOTYPES)}] {phenotype}: {answer_text} (prob: {probability:.4f}, logprob: {logprob:.4f})")
            else:
                # Fallback if no logprobs available
                prediction = 1 if "yes" in answer_text.lower() else 0
                results.append({
                    'phenotype': phenotype,
                    'prediction': prediction,
                    'logprob': None,
                    'probability': None
                })
                print(f"[{i+1}/{len(PHENOTYPES)}] {phenotype}: {answer_text} (no logprobs available)")

        except Exception as e:
            print(f"Error classifying phenotype '{phenotype}': {e}")
            results.append({
                'phenotype': phenotype,
                'prediction': 0,
                'logprob': None,
                'probability': None
            })

    return results

def classify_phenotypes_local(diagnosis_answer, tokenizer, model):
    """
    Classify which phenotypes the patient has using local model.
    Uses per-phenotype classification to get logprobs for each phenotype.

    Args:
        diagnosis_answer: The diagnosis text from <answer> tags
        tokenizer: The tokenizer
        model: The LLM model

    Returns:
        List of dictionaries with phenotype name, prediction (0 or 1), and logprob
    """
    import math
    import torch.nn.functional as F
    results = []

    for i, phenotype in enumerate(PHENOTYPES):
        prompt = f"""Based on the following diagnosis, does the patient have "{phenotype}"?

Diagnosis:
{diagnosis_answer}

Answer only "Yes" or "No"."""

        if tokenizer.chat_template:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False
            )

        try:
            input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
            attention_mask = torch.ones_like(input_ids)

            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=5,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
                temperature=0.0,
                output_scores=True,
                return_dict_in_generate=True
            )

            generated_tokens = outputs.sequences[0][input_ids.shape[1]:]
            answer_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

            # Get logprobs from the first generated token
            if len(outputs.scores) > 0:
                first_token_scores = outputs.scores[0][0]  # Shape: [vocab_size]
                first_token_logprobs = F.log_softmax(first_token_scores, dim=-1)

                # Get the actual generated token
                first_generated_token_id = generated_tokens[0].item()
                first_token_logprob = first_token_logprobs[first_generated_token_id].item()

                # Try to find Yes/No token logprobs
                yes_tokens = tokenizer.encode("Yes", add_special_tokens=False)
                no_tokens = tokenizer.encode("No", add_special_tokens=False)

                yes_logprob = first_token_logprobs[yes_tokens[0]].item() if yes_tokens else None
                no_logprob = first_token_logprobs[no_tokens[0]].item() if no_tokens else None

                # Determine prediction and logprob
                if "yes" in answer_text.lower():
                    prediction = 1
                    logprob = yes_logprob if yes_logprob is not None else first_token_logprob
                else:
                    prediction = 0
                    logprob = no_logprob if no_logprob is not None else first_token_logprob

                # Convert logprob to probability for display
                probability = math.exp(logprob)

                results.append({
                    'phenotype': phenotype,
                    'prediction': prediction,
                    'logprob': logprob,
                    'probability': probability
                })

                print(f"[{i+1}/{len(PHENOTYPES)}] {phenotype}: {answer_text} (prob: {probability:.4f}, logprob: {logprob:.4f})")
            else:
                # Fallback if no scores available
                prediction = 1 if "yes" in answer_text.lower() else 0
                results.append({
                    'phenotype': phenotype,
                    'prediction': prediction,
                    'logprob': None,
                    'probability': None
                })
                print(f"[{i+1}/{len(PHENOTYPES)}] {phenotype}: {answer_text} (no logprobs available)")

        except Exception as e:
            print(f"Error classifying phenotype '{phenotype}': {e}")
            results.append({
                'phenotype': phenotype,
                'prediction': 0,
                'logprob': None,
                'probability': None
            })

    return results

def generate_report(episode_id, question, results, documents):
    """Generate a detailed report of the diagnosis."""
    report = []
    report.append("=" * 80)
    report.append(f"MEDICAL DIAGNOSIS REPORT - EPISODE {episode_id}")
    report.append("=" * 80)
    report.append(f"\nQuestion: {question}")
    report.append(f"\nTotal document pieces available: {len(documents)}")
    report.append(f"Number of searches performed: {results['num_searches']}")

    # Initial summary
    if results.get('initial_summary'):
        report.append("\n" + "-" * 80)
        report.append("INITIAL SUMMARY (from last section of note):")
        report.append("-" * 80)
        report.append(results['initial_summary'])

    # Search history
    if results['search_history']:
        report.append("\n" + "-" * 80)
        report.append("SEARCH QUERIES:")
        report.append("-" * 80)
        for idx, query in enumerate(results['search_history'], 1):
            report.append(f"{idx}. {query}")

    # Retrieved passages
    if results['retrieved_passages']:
        report.append("\n" + "-" * 80)
        report.append(f"RETRIEVED PASSAGES (Top {TOPK_RETRIEVAL} per query):")
        report.append("-" * 80)
        seen_ids = set()
        for idx, passage in enumerate(results['retrieved_passages'], 1):
            doc_id = passage['document']['id']
            if doc_id not in seen_ids:  # Avoid duplicates
                seen_ids.add(doc_id)
                content = passage['document']['contents']
                score = passage.get('score', 'N/A')
                report.append(f"\nPassage {idx} (ID: {doc_id}, Score: {score:.2f}):")
                report.append(content)

    # Reasoning blocks
    if results['reasoning_blocks']:
        report.append("\n" + "-" * 80)
        report.append("LLM REASONING:")
        report.append("-" * 80)
        for idx, reasoning in enumerate(results['reasoning_blocks'], 1):
            report.append(f"\nReasoning block {idx}:")
            report.append(reasoning.strip())

    # Final answer
    report.append("\n" + "=" * 80)
    report.append("FINAL DIAGNOSIS:")
    report.append("=" * 80)
    if results['final_answer']:
        report.append(results['final_answer'])
    else:
        report.append("No answer found in <answer> tags")

    # Classified phenotypes with logprobs
    if results.get('classified_phenotypes') is not None:
        report.append("\n" + "=" * 80)
        report.append("CLASSIFIED PHENOTYPES WITH LOGPROBS:")
        report.append("=" * 80)

        classified_phenotypes = results['classified_phenotypes']

        # Check if new format (list of dicts with logprobs)
        if classified_phenotypes and isinstance(classified_phenotypes[0], dict):
            # Separate positive and negative predictions
            positive_phenotypes = [p for p in classified_phenotypes if p['prediction'] == 1]
            negative_phenotypes = [p for p in classified_phenotypes if p['prediction'] == 0]

            if positive_phenotypes:
                report.append("\nPOSITIVE PREDICTIONS:")
                report.append("-" * 40)
                for idx, p in enumerate(positive_phenotypes, 1):
                    prob_str = f"prob: {p['probability']:.4f}" if p['probability'] is not None else "prob: N/A"
                    logprob_str = f"logprob: {p['logprob']:.4f}" if p['logprob'] is not None else "logprob: N/A"
                    report.append(f"{idx}. {p['phenotype']}")
                    report.append(f"   {prob_str}, {logprob_str}")
            else:
                report.append("\nNo positive phenotype predictions")

            report.append("\n" + "-" * 80)
            report.append("ALL PHENOTYPES (sorted by probability):")
            report.append("-" * 80)

            # Sort by probability (highest first)
            sorted_phenotypes = sorted(
                classified_phenotypes,
                key=lambda x: x['probability'] if x['probability'] is not None else -float('inf'),
                reverse=True
            )

            for idx, p in enumerate(sorted_phenotypes, 1):
                pred_str = "YES" if p['prediction'] == 1 else "NO "
                prob_str = f"{p['probability']:.4f}" if p['probability'] is not None else "N/A   "
                logprob_str = f"{p['logprob']:+.4f}" if p['logprob'] is not None else "N/A   "
                report.append(f"{idx:2d}. [{pred_str}] {p['phenotype']:<70s} prob: {prob_str}  logprob: {logprob_str}")

        # Legacy format (list of strings)
        elif classified_phenotypes and isinstance(classified_phenotypes[0], str):
            for idx, phenotype in enumerate(classified_phenotypes, 1):
                report.append(f"{idx}. {phenotype}")
        else:
            report.append("No phenotypes matched from the predefined list")

    report.append("\n" + "=" * 80)

    return "\n".join(report)

def main():
    # Load data
    print("Starting medical diagnosis inference...")
    import pandas as pd
    data = load_data(DATA_PATH)

    # Get first episode for testing
    # Assuming data is a dict with episode IDs as keys, a list, or a DataFrame
    if isinstance(data, pd.DataFrame):
        # For DataFrame, get the first row
        # Assuming there's an index or first column as episode_id and second column as concatenated note
        first_row = data.iloc[0]
        episode_id = data.iloc[0]['EPISODE_ID']
        # Try to find the concatenated note - it's likely the first or second column
        # or a column with 'note' or 'text' in the name
        note_columns = [col for col in data.columns if 'note' in col.lower() or 'text' in col.lower()]
        if note_columns:
            concatenated_note = first_row[note_columns[0]]
        else:
            # Assume it's the first column or last column
            concatenated_note = first_row.iloc[0] if len(first_row) > 0 else first_row.iloc[-1]
        print(f"Using DataFrame with {len(data)} episodes")
        print(f"DataFrame columns: {list(data.columns)}")
    elif isinstance(data, dict):
        episode_id = list(data.keys())[0]
        concatenated_note = data[episode_id]
    elif isinstance(data, list):
        episode_id = 0
        concatenated_note = data[0]
    else:
        raise ValueError(f"Unexpected data format: {type(data)}")

    print(f"\nProcessing episode: {episode_id}")
    print(f"Note length: {len(concatenated_note)} characters")

    # Split and format documents
    documents = split_and_format_documents(concatenated_note, episode_id)

    # Extract and summarize the last section of the note
    print("\n" + "=" * 80)
    print("STEP 1: Extracting and summarizing last section of note")
    print("=" * 80)
    last_section = extract_last_section(concatenated_note)

    if last_section:
        print(f"Last section extracted ({len(last_section)} characters)")
        print(f"First 200 characters: {last_section[:200]}...")
    else:
        print("No separator found - no last section to extract")

    # Define the diagnosis question
    question = "Diagnose all conditions that happened during this patient's hospitalization stay."

    # Run inference based on mode
    if USE_VLLM:
        print(f"\nUsing vLLM server: {VLLM_SERVER_URL}")
        print(f"Model: {MODEL_ID}")

        # Summarize last section
        if last_section:
            print("\nSummarizing last section...")
            initial_summary = summarize_last_section_vllm(last_section)
            print(f"\nSummary of last section:\n{initial_summary}")
        else:
            initial_summary = ""

        print("\n" + "=" * 80)
        print("STEP 2: Starting diagnosis with search and reasoning")
        print("=" * 80)
        results = run_inference_vllm(question, documents, initial_summary=initial_summary)

        # Classify phenotypes if we have a final answer
        if results['final_answer']:
            print('\n\n################# [Phenotype Classification] ##################\n')
            print(f"Classifying phenotypes based on diagnosis answer...")
            classified_phenotypes = classify_phenotypes_vllm(results['final_answer'])
            results['classified_phenotypes'] = classified_phenotypes

            # Count positive predictions
            positive_count = sum(1 for p in classified_phenotypes if p['prediction'] == 1)
            print(f"\nIdentified {positive_count} positive phenotype(s):")
            for p in classified_phenotypes:
                if p['prediction'] == 1:
                    prob_str = f"prob: {p['probability']:.4f}" if p['probability'] is not None else "prob: N/A"
                    print(f"  - {p['phenotype']} ({prob_str})")
        else:
            results['classified_phenotypes'] = []
            print("\nNo final answer found - skipping phenotype classification")
    else:
        # Initialize model and tokenizer
        print(f"\nInitializing local model: {MODEL_ID}")
        print(f"Device: {device}")
        tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map=0
        )

        # Summarize last section
        if last_section:
            print("\nSummarizing last section...")
            initial_summary = summarize_last_section_local(last_section, tokenizer, model)
            print(f"\nSummary of last section:\n{initial_summary}")
        else:
            initial_summary = ""

        # Initialize stopping criteria
        target_sequences = ["</search>", " </search>", "</search>\n", " </search>\n", "</search>\n\n", " </search>\n\n"]
        stopping_criteria = transformers.StoppingCriteriaList([StopOnSequence(target_sequences, tokenizer)])

        print("\n" + "=" * 80)
        print("STEP 2: Starting diagnosis with search and reasoning")
        print("=" * 80)
        # Run inference
        results = run_inference(question, documents, tokenizer, model, stopping_criteria, initial_summary=initial_summary)

        # Classify phenotypes if we have a final answer
        if results['final_answer']:
            print('\n\n################# [Phenotype Classification] ##################\n')
            print(f"Classifying phenotypes based on diagnosis answer...")
            classified_phenotypes = classify_phenotypes_local(results['final_answer'], tokenizer, model)
            results['classified_phenotypes'] = classified_phenotypes

            # Count positive predictions
            positive_count = sum(1 for p in classified_phenotypes if p['prediction'] == 1)
            print(f"\nIdentified {positive_count} positive phenotype(s):")
            for p in classified_phenotypes:
                if p['prediction'] == 1:
                    prob_str = f"prob: {p['probability']:.4f}" if p['probability'] is not None else "prob: N/A"
                    print(f"  - {p['phenotype']} ({prob_str})")
        else:
            results['classified_phenotypes'] = []
            print("\nNo final answer found - skipping phenotype classification")

    # Generate and print report
    report = generate_report(episode_id, question, results, documents)
    print("\n\n")
    print(report)

    # Optionally save report to file
    output_file = f"diagnosis_report_episode_{episode_id}.txt"
    with open(output_file, 'w') as f:
        f.write(report)
    print(f"\nReport saved to: {output_file}")

if __name__ == "__main__":
    main()
