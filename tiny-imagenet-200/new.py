import wikipedia
import random

def generate_quiz_questions(num_questions):
    questions = []
    for _ in range(num_questions):
        page = wikipedia.random(1)
        try:
            summary = wikipedia.summary(page, sentences=1)
            options = [page] + random.sample(wikipedia.random(3), 3)
            random.shuffle(options)
            questions.append({
                'question': f'What/who is {summary}?',
                'options': options,
                'answer': page
            })
        except wikipedia.exceptions.DisambiguationError as e:
            continue
    return questions

# Generate 10 quiz questions
quiz_questions = generate_quiz_questions(10)
for i, question in enumerate(quiz_questions, 1):
    print(f'Question {i}: {question["question"]}')
    for j, option in enumerate(question['options'], 1):
        print(f'    Option {j}: {option}')
    print(f'    Answer: {question["answer"]}\n')
