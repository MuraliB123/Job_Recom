import json

from collections import Counter

data = {}

frequency_role = {}

employment_types_mapping = {
    'Internship' : 1,
    'Part-time' : 2,
    'Contract'  : 3,
    'Full-time' : 4,
    'Temporary' : 5,
    'Other' : 6
}
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

# Load the dataset
def load_data():
    global data
    with open('assignment_test_job.json', 'r') as file:
        data = json.load(file)


def extract_and_assign(data):
    """

     Transform job_functions_collection column to extract only the functional role associated with the job.
     Transform job_industries_collection column to extract only the industries associated with the job.
     Delete the columns - job_status_log_collection and few others as below

    """
    for job in data:
        functional_roles = []
        for item in job["job_functions_collection"]:
            function = item["job_function_list"]["function"]
            functional_roles.append(function)
        if len(functional_roles) > 0:
            job["job_functions_collection"] = functional_roles[0]
        else:
            job["job_functions_collection"] = "N/A"

        industries = []
        for item in job["job_industries_collection"]:
            industry = item["job_industry_list"]["industry"]
            industries.append(industry)
        job["job_industries_collection"] = industries
        
        del job['job_status_log_collection']
        del job['_id']
        del job['linkedin_job_id']
        del job['last_updated_ux']
        del job['redirected_url_hash']
        del job['hash']
        del job['application_active']
        del job['external_url']
        del job['redirected_url']


def transform(data):
    """
    job_functions_collection has words with commans and 'and'. do split on them and make a list of splitted words for each job.
    """
    for item in data:
        item['job_functions_collection'] = [job_function.replace('and ', '').strip() for job_function in item['job_functions_collection'].split(',')]


def get_max_frequency(job_functions):
    temp = max(frequency_role.get(role, 0) for role in job_functions)
    return temp


def sort_data(data):
    """
    Sort the data by job_role's frequency such that ordering is introduced in the data. 
    Jobs of similar functional_role will be closer to each other.
    """
    global frequency_role 
    functional_roles = []
    for item in data:
        for role in item['job_functions_collection']:
            functional_roles.append(role)
    word_counts = Counter(functional_roles)

    global frequency_role 

    for word, count in word_counts.items():
        frequency_role[word] = count
    
    data = sorted(data, key=lambda x: get_max_frequency(x['job_functions_collection']), reverse=True)


# Save the transformed data
def save_data(data, filename):
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)


if __name__ == '__main__':
    load_data()
    extract_and_assign(data)
    transform(data)
    sort_data(data)
    save_data(data, 'dataset.json')