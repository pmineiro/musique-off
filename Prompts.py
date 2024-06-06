def format_fact(fact):
    return f"Title: {fact['title']}\nContent: {fact['paragraph_text']}"

def format_knowledge(knowledge):
    return '\n'.join([ format_fact(fact) for fact in knowledge ])

def format_for_query(*, question, knowledge, ctsig):
    system = f"You are a query writing agent.  Given a Question and Current Documents, formulate a simple natural language query for retrieval which would retrieve a relevant New Document.  Be as brief as possible.  Do not explain your reasoning.  Output only the simple query."
    user = f"Original Question:\n{question}\n\nCurrent Documents:\n{format_knowledge(knowledge)}"

    return ctsig.render_with_system(system=system, user=user)

def format_for_ranking(*, question, knowledge, used, ctsig):
    current_documents = format_knowledge([ fact for n, fact in enumerate(knowledge) if n in used ])

    system = f"In the following task, you are given a Question, some Current Documents, and a New Document. You should decide whether the New Document contains relevant additional information for answering the Question."

    msgs = []
    orig_indices = []
    for n, fact in enumerate(knowledge):
        if n not in used:
            orig_indices.append(n)
            user = f"Question:\n{question}\n\nCurrent Documents:\n{current_documents}\n\nNew Document:\n{format_fact(fact)}\n\nDoes the New Document contain relevant additional information for answering the Question?  Output only Yes or No."
            msgs.append(ctsig.render_with_system(system=system, user=user))

    return msgs, orig_indices

def format_for_stop_retrieval(*, question, knowledge, ctsig):
    system = f"In the following task, you are given a Question and Current Documents. You should decide whether the Current Documents are sufficient to answer the Question."
    user = f"Question:\n{question}\n\nCurrent Documents:\n{format_knowledge(knowledge)}\n\nIs there enough information in the Current Documents to answer the Question?  Output only Yes or No."

    return ctsig.render_with_system(system=system, user=user)

def format_for_intermediate(*, question, knowledge, ctsig):
    question = ' '.join(question.split())
    system = f'Answer the Question using the Documents. The Documents may be insufficient or irrelevant.'
    user = f"Question:\n{question}\n\nDocuments:\n{format_knowledge(knowledge)}"
    return ctsig.render_with_system(system=system, user=user)

def format_for_final(*, question, guess, ctsig):
    guess = ' '.join(guess.split())
    system = f'You are given a Question and an Verbose Answer.  Summarize the Verbose Answer as briefly as possible.  Do not explain your reasoning.'
    user = f'Question:\n{question}\n\nVerbose Answer:\n{guess}'

    return ctsig.render_with_system(system=system, user=user)

def format_for_sufficiency(*, question, knowledge, ctsig):
    system = f"In the following task, you are given a Question and Current Documents. You should decide whether the Current Documents are sufficient to answer the Question."
    user = f"Question:\n{question}\n\nCurrent Documents:\n{format_knowledge(knowledge)}\n\nIs there enough information in the Current Documents to answer the Question?  Output only Yes or No."

    return ctsig.render_with_system(system=system, user=user)
