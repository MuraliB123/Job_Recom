from flask import Flask, render_template, request
import json
import pandas as pd
from collections import Counter
import json
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from Data_processing import frequency_role
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

seniority_level_mapping = {

    "Internship": 1,
    "Entry level": 2,
    "Mid-Senior level": 3,
    "Associate": 4,
    "Executive":5,
    "Director": 6,
    "Not Applicable": 7,
    '': 8

}

seniority_levels_dict = [
    "Internship", "Entry level", "Mid-Senior level",
    "Associate", "Executive", "Director", "Not Applicable"
]

employment_types_dict = [
    'Full-time', 'Contract', 'Temporary', 'Other', 'Part-time', 'Internship'
]

roles_of_interest = [
    'Engineering Information Technology', 'Business Development Sales', 'Management Manufacturing', 
    'Sales Business Development', 'Accounting/Auditing Finance', 'Information Technology', 
    'Design', 'Art/Creative', 'Marketing Sales', 'N/A', 'Other', 'Engineering', 'Legal', 
    'Purchasing Supply Chain', 'Finance Sales', 'Human Resources', 'Sales', 'Research', 
    'Analyst', 'Consulting', 'Education Training', 'Administrative', 'Project Management Information Technology', 
    'Information Technology Engineering', 'Project Management', 'Business Development', 
    'Strategy/Planning Consulting', 'Marketing', 'Public Relations', 'Writing/Editing', 'Finance', 
    'Health Care Provider', 'Sales Management', 'Production', 'General Business', 'Quality Assurance', 
    'Administrative Customer Service', 'Accounting/Auditing', 'Administrative Writing/Editing', 
    'Project Management Management', 'Administrative Engineering', 'Customer Service', 'Education', 
    'Business Development Finance', 'Strategy/Planning Information Technology', 'Legal Science', 
    'Management', 'Strategy/Planning', 'Information Technology Design', 'Marketing Project Management', 
    'Science', 'Product Management Marketing', 'Research Engineering', 'Training', 
    'Research Education', 'Supply Chain', 'Product Management', 'Human Resources Management', 
    'Advertising', 'Customer Service Engineering', 'Information Technology Other', 'Consulting Other', 
    'Product Management Management', 'Analyst Finance', 'Project Management Analyst', 
    'Information Technology Art/Creative', 'Human Resources Information Technology', 
    'Research Analyst', 'Legal Other', 'Distribution', 'Administrative Information Technology', 
    'Sales Strategy/Planning', 'Research Strategy/Planning', 'Art/Creative Information Technology', 
    'Quality Assurance Analyst', 'Project Management Customer Service', 
    'Information Technology Human Resources', 'Quality Assurance Project Management', 
    'Quality Assurance Customer Service', 'Marketing Management', 'Engineering Analyst', 
    'Design Art/Creative', 'Information Technology Management', 'Purchasing', 'Human Resources Administrative', 
    'Other Manufacturing', 'Information Technology Finance', 'Sales Marketing', 'Manufacturing', 
    'Quality Assurance Engineering', 'Information Technology Customer Service', 'Marketing Art/Creative', 
    'Information Technology Science', 'Health Care Provider Other', 'Information Technology Administrative', 
    'Customer Service Other', 'Engineering Quality Assurance', 'Health Care Provider Human Resources', 
    'Consulting Customer Service', 'Analyst Other', 'Production Writing/Editing', 
    'Information Technology Consulting', 'Information Technology Supply Chain', 
    'Information Technology Project Management', 'Consulting Information Technology', 'Research Design', 
    'Finance Consulting', 'Finance Accounting/Auditing', 'Product Management Engineering', 
    'Information Technology Quality Assurance', 'Administrative Project Management', 
    'Administrative Management', 'Production Manufacturing', 'Customer Service Sales', 
    'Business Development Information Technology', 'Administrative Strategy/Planning', 
    'Advertising Marketing', 'Marketing Advertising', 'Marketing Other', 'Administrative General Business', 
    'Design Consulting', 'Education Writing/Editing', 'Finance Management', 'Business Development Education', 
    'Quality Assurance Marketing', 'Information Technology Product Management', 'Finance Analyst', 
    'Management Human Resources', 'Other Writing/Editing', 'Information Technology Analyst', 
    'Writing/Editing Art/Creative', 'Research Consulting'
]

recom_jobs=[] 


employment_types_mapping = {
    'Internship' : 1,
    'Part-time' : 2,
    'Contract'  : 3,
    'Full-time' : 4,
    'Temporary' : 5,
    'Other' : 6
}

data = {}

inverse_job_role = {}

inverse_seniority = {}

inverse_employment = {}

job_function_to_label = {}
relevant_jobs = []
non_relevant_jobs = []

nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
inputs = []
max_length = 0

X = None

def get_frequency(job_function):
    return frequency_role.get(job_function, 0)

def create_encoding(data):
    """
     encode 'job_functions_collection' 'seniority' 'employment type' and replace original dataset
    """
    global job_function_to_label

    all_job_functions = set()

    for job in data:
        all_job_functions.update(job['job_functions_collection'])

    sorted_job_functions = sorted(all_job_functions, key=get_frequency, reverse=True)

    job_function_to_label = {job_function: idx for idx, job_function in enumerate(sorted_job_functions)}

    for job in data:
        job['job_functions_collection'] = [job_function_to_label[job_function] for job_function in job['job_functions_collection']]

    for job in data:
        job["seniority"] = seniority_level_mapping[job["seniority"]]

    for job in data:
        job["employment_type"] = employment_types_mapping[job["employment_type"]]



def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and special characters
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    # Remove stopwords and lemmatize
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def encode_job(job,max_length):
    seniority = job["seniority"]
    employment_type = job["employment_type"]
    job_functions = job["job_functions_collection"]

    # Pad job functions to max_length
    job_functions_padded = job_functions + [-1] * (max_length - len(job_functions))

    return [seniority, employment_type] + job_functions_padded
    
def load_model():
    global max_length
    max_length = max(len(job["job_functions_collection"]) for job in data)
    features = []
    for job in data:
        features.append(encode_job(job,max_length))
    X = np.array(features)
    return X

@app.route('/')
def index():
    global data
    with open('dataset.json', 'r') as file:
        data = json.load(file)

    create_encoding(data)
    return render_template('index.html', seniority_levels=seniority_levels_dict, employment_types=employment_types_dict, roles_of_interest=roles_of_interest)

@app.route('/submit', methods=['POST'])
def submit():
    """ 
    Main utility function which takes input features and output recommendations using KNN
    """
    global inverse_job_role,inverse_seniority,inverse_employment,recom_jobs,inputs,X
    
    seniority_level = request.form.get('seniority_level')
    employment_type = request.form.get('employment_type')
    selected_roles = request.form.getlist('roles_of_interest')

    X =  load_model()

    input_features = [seniority_level_mapping[seniority_level], employment_types_mapping[employment_type]]


    for job_function in selected_roles:
        input_features.append(job_function_to_label[job_function])

    input_features = input_features + [-1] * (max_length - len(selected_roles))
    inputs = input_features

    knn = KNeighborsClassifier(n_neighbors=20)


    knn.fit(X, np.zeros(len(X))) 


    neighbors = knn.kneighbors([input_features], return_distance=False)

    nearest_jobs = [data[idx] for idx in neighbors[0]]


    for key,value in job_function_to_label.items():
        inverse_job_role[value] = key


    
    for key,value in seniority_level_mapping.items():
        inverse_seniority[value] = key

    for key,value in employment_types_mapping.items():
        inverse_employment[value] = key


    for job in nearest_jobs:
        job["seniority"] = inverse_seniority[job["seniority"]]
        job["employment_type"] = inverse_employment[job["employment_type"]]
        job["job_functions_collection"] = [inverse_job_role[job_function] for job_function in job["job_functions_collection"]]

    recom_jobs = nearest_jobs

    return render_template('results.html', jobs=nearest_jobs, seniority_level=seniority_level, employment_type=employment_type, selected_roles=selected_roles)
 
@app.route('/search_gpt', methods=['POST', 'GET'])
def query():
    """
    Vectorise each job's description using TF-IDF and retrive the similar jobs based on user query
    """
    if request.method == 'POST':
        global data, inverse_job_role, inverse_seniority, inverse_employment
    
    # Retrieve job descriptions and preprocess them
        for job in data:
            job["description"] = preprocess_text(job["description"])
    
        job_descriptions = [job["description"] for job in data]

    # Vectorize job descriptions using TF-IDF
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(job_descriptions)

    # Retrieve input job description from the HTML form
        input_description = request.form.get('job_description', '')

    # Preprocess the input description
        input_description = preprocess_text(input_description)

    # Vectorize the input description
        input_vector = vectorizer.transform([input_description])

    # Calculate cosine similarity between input vector and job descriptions
        similarity_scores = cosine_similarity(input_vector, X).flatten()

    # Retrieve top N jobs based on similarity scores
        N = 5
        top_indices = similarity_scores.argsort()[-N:][::-1]
        relevant_jobs = [data[idx] for idx in top_indices]

    # Convert label indices back to human-readable formats
        for job in relevant_jobs:
            job["seniority"] = inverse_seniority[job["seniority"]]
            job["employment_type"] = inverse_employment[job["employment_type"]]
            job["job_functions_collection"] = [inverse_job_role[job_function] for job_function in job["job_functions_collection"]]

    # Render the results using a results.html template
        return render_template('results.html', jobs=relevant_jobs)
    
    return render_template('search.html')

   
def rocchio_update_query_vector(input_features, relevant_jobs, non_relevant_jobs, alpha=1, beta=0.75, gamma=0.25):
    """
    Update and return the new query vector using Rocchio Relevance Feedback.

    Parameters:
    - input_features (list): The initial query vector.
    - relevant_jobs (list of lists): The vectors of relevant jobs.
    - non_relevant_jobs (list of lists): The vectors of non-relevant jobs.
    - alpha (float): Weight for the original query vector.
    - beta (float): Weight for the relevant job vectors.
    - gamma (float): Weight for the non-relevant job vectors.

    Returns:
    - new_query_vector (list): The updated query vector.
    """
    # Convert input features and job vectors to numpy arrays
    input_vector = np.array(input_features)
    relevant_vectors = np.array(relevant_jobs)
    non_relevant_vectors = np.array(non_relevant_jobs)

    # Calculate mean vectors for relevant and non-relevant jobs
    if len(relevant_vectors) > 0:
        mean_relevant_vector = relevant_vectors.mean(axis=0)
    else:
        mean_relevant_vector = np.zeros_like(input_vector)

    if len(non_relevant_vectors) > 0:
        mean_non_relevant_vector = non_relevant_vectors.mean(axis=0)
    else:
        mean_non_relevant_vector = np.zeros_like(input_vector)

    # Apply Rocchio formula to adjust the query vector
    new_query_vector = (alpha * input_vector +
                        beta * mean_relevant_vector -
                        gamma * mean_non_relevant_vector)

    return new_query_vector.tolist()

@app.route('/feedback', methods=['POST'])
def feedback():
    global relevant_jobs, non_relevant_jobs

    job_id = request.form.get('job_id')
    sl = request.form.get('sl')
    et = request.form.get('et')
    roles = request.form.getlist('roles')

    job_vector = [seniority_level_mapping[sl], employment_types_mapping[et]]

    for job_function in roles:
        job_vector.append(job_function_to_label[job_function])

    job_vector = job_vector + [-1] * (max_length - len(roles))

    feedback_type = request.form.get('feedback')
    if feedback_type == "relevant":
        relevant_jobs.append(job_vector)
    elif feedback_type == "not_relevant":
        non_relevant_jobs.append(job_vector)

    # Redirect back to the results page 
    return render_template('results.html',jobs=recom_jobs)


@app.route('/refresh') 
def update_query():
    global inverse_job_role,inverse_seniority,inverse_employment

    new_query = rocchio_update_query_vector(inputs,relevant_jobs,non_relevant_jobs)
    
    knn = KNeighborsClassifier(n_neighbors=20)

   # print(type(new_query))
    knn.fit(X, np.zeros(len(X))) 


    neighbors = knn.kneighbors([new_query], return_distance=False)

    nearest_jobs = [data[idx] for idx in neighbors[0]]


    for job in nearest_jobs:
        if type(job["seniority"]) != str:
            job["seniority"] = inverse_seniority[job["seniority"]]
        if type(job["employment_type"]) != str:
            job["employment_type"] = inverse_employment[job["employment_type"]]
        if any(type(func) != str for func in job["job_functions_collection"]):
            job["job_functions_collection"] = [inverse_job_role[job_function] for job_function in job["job_functions_collection"]]
   
    return render_template('results.html', jobs=nearest_jobs)
    


if __name__ == '__main__':
    app.run(debug=True)
