# The necessary imports for this application
import requests
import json
import os

from time import sleep


download_after_explain = True

### This costs lots of money $$$$ to run
### To avoid accidents, we did this.
#print ("Not to be run unless approved in advance")
#exit(0)

# The URL of the Extractions Endpoint
url = 'https://api.dowjones.com/alpha/extractions/documents'

# Our prompts to be inserted into our query.

headers = {'content-type': 'application/json', 'user-key': 'b26c35f58622e4064bcd5b492d841193'}

request_body = {
  "query": {
    "where" : "(REGEXP_CONTAINS(CONCAT(title, ' ', IFNULL(snippet, ''), ' ', IFNULL(body, '')), r'(?i)(\\b)(protest\\w{0,}|demonstration\\w{0,}|riot\\w{0,})(\\b)') "
              "AND language_code='en' AND publication_date >= '2010-01-01 00:00:00' AND LOWER(source_code) IN ('aprs','lba','xnews','afnws','ajazen','bbcsup','bbcapp','bbcmep','bbceup','bbcap','bbcca','bbcmnf','bbcmap','bbcukb','bbcukb','bbcsap','bbccau','bbcmm'))",
      "includes": {
    	"subject_codes": ["gcrim","gglobe","gcns","gpir","grisk","gvio","gcoup","gsec","gspy","gdef","gwar"],
        "region_codes": ["africaz","asiaz","eeurz","lamz"]
    }
  }
}

# Create an explain with the given query
print("Creating an explain: " + json.dumps(request_body))
response = requests.post(url + "/_explain", data=json.dumps(request_body), headers=headers)

# Check the explain to verify the query was valid and see how many docs would be returned
if response.status_code != 201:
    print("ERROR: An error occurred creating an explain: " + response.text)
else:
    explain = response.json()
    print("Explain Created. Job ID: " + explain["data"]["id"])
    state = explain["data"]["attributes"]["current_state"]

    # wait for explain job to complete
    while state != "JOB_STATE_DONE":
        self_link = explain["links"]["self"]
        response = requests.get(self_link, headers=headers)
        explain = response.json()
        state = explain["data"]["attributes"]["current_state"]

    print("Explain Completed Successfully.")
    doc_count = explain["data"]["attributes"]["counts"]
    print("Number of documents returned: " + str(doc_count))

    if not download_after_explain:
        print("Not proceeding with extraction, since not wanted. Exiting.")
    else:
        print("Sleeping 5 seconds before extraction")
        sleep(5)
        # Create a snapshot with the given query
        print("Creating the snapshot: " + json.dumps(request_body))
        response = requests.post(url, data=json.dumps(request_body), headers=headers)
        print(response.text)

        # Verify the response from creating an extraction is OK
        if response.status_code != 201:
            print("ERROR: An error occurred creating an extraction: " + response.text)
        else:
            extraction = response.json()
            print(extraction)
            print("Extraction Created. Job ID: " + extraction['data']['id'])
            self_link = extraction["links"]["self"]
            sleep(30)
            print ("Checking state of the job...")

            while True:
                # We now call the second endpoint, which will tell us if the extraction is ready.
                status_response = requests.get(self_link, headers=headers)

                # Verify the response from the self_link is OK
                if status_response.status_code != 200:
                    print("ERROR: an error occurred getting the details for the extraction: " + status_response.text)
                else:
                    # There is an edge case where the job does not have a current_state yet. If current_state
                    # does not yet exist in the response, we will sleep for 10 seconds
                    status = status_response.json()

                    if 'current_state' in status['data']['attributes']:
                        currentState = status['data']['attributes']['current_state']
                        print("Current state is: " + currentState)

                        # Job is still running, Sleep for 10 seconds
                        if currentState == "JOB_STATE_RUNNING":
                            print("Sleeping for 30 seconds... Job state running")
                            sleep(30)

                        elif currentState == "JOB_VALIDATING":
                            print("Sleeping for 30 seconds... Job validating")
                            sleep(30)

                        elif currentState == "JOB_QUEUED":
                            print("Sleeping for 30 seconds... Job queued")
                            sleep(30)

                        elif currentState == "JOB_CREATED":
                            print("Sleeping for 30 seconds... Job created")
                            sleep(30)

                        else:
                            # If currentState is JOB_STATE_DONE then everything completed successfully
                            if currentState == "JOB_STATE_DONE":
                                print("Job completed successfully")
                                print("Downloading snapshot files to current directory")
                                for file in status['data']['attributes']['files']:
                                    filepath = file['uri']
                                    parts = filepath.split('/')
                                    filename = parts[len(parts) - 1]
                                    r = requests.get(file['uri'], stream=True, headers=headers)
                                    dir_path = os.path.dirname(os.path.realpath(__file__))
                                    filename = os.path.join(dir_path, filename)
                                    with open(filename, 'wb') as fd:
                                        for chunk in r.iter_content(chunk_size=128):
                                            fd.write(chunk)

                            # job has another state that means it was not successful.
                            else:
                                print("An error occurred with the job. Final state is: " + currentState)

                            break
                    else:
                        print("Sleeping for 30 seconds...")
                        sleep(30)
