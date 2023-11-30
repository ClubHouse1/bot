from flask import Flask, render_template, request, jsonify
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

answers_dir_path = os.path.dirname(os.path.abspath(__file__))
answers_file_path = os.path.join(answers_dir_path, "answers.txt")
answers = {}

def read_answers_file(file_path):
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if line:
                parts = line.split(":")
                if len(parts) >= 2:
                    key = parts[0].strip()
                    values = parts[1].split(",")
                    answers[key] = [value.strip() for value in values]

if os.path.isfile(answers_file_path):
    read_answers_file(answers_file_path)

app = Flask(__name__)

@app.route('/ask', methods=['GET', 'POST'])
def ask():
    if request.method == 'POST':
        user_query = request.form['user_query']

        if user_query.lower() == "exit":
            os._exit(0)  # Stop the Flask development server

        if not answers:
            return jsonify({'response': "No questions available. Please add questions to your answers.txt file."})
        else:
            vectorizer = TfidfVectorizer()
            question_vectors = vectorizer.fit_transform(list(answers.keys()) + [user_query])
            similarity_scores = cosine_similarity(question_vectors[:-1], question_vectors[-1])
            most_similar_index = similarity_scores.argmax()
            most_similar_question = list(answers.keys())[most_similar_index]

            compatible_answers = answers[most_similar_question]
            output = compatible_answers[0]

            if len(compatible_answers) > 1:
                max_similarity = similarity_scores[most_similar_index].max()
                best_answer_index = similarity_scores[most_similar_index].argmax()
                output = compatible_answers[best_answer_index]

            return jsonify({'response': output})

    # Handle GET request, if needed
    return render_template('your_form_template.html')

if __name__ == '__main__':
    app.run(debug=True)
