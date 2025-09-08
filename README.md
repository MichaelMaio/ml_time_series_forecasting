# Production-Ready, Reproducible, Secure, Cross-Cloud Machine Learning Pipeline #
Engineer: Michael Maio<br>
Last updated: 9/7/2025

## Overview ##
This repo contains a working machine learning pipeline that addresses the following hypothetical scenario that a software company might need to deal with.<br>

**Problem:** Data center power consumption is growing, causing a gradual, year-over-year increase in the hourly peak kilowatt load reported by a sensor on the local transformer.<br><br>
**Question:** Assuming the power consumption trend remains unchanged, how long before the transformer becomes overloaded?<br><br>
**Solution:** Build a machine learning pipeline that can process recent trend data and forecast when the transformer may eventually become overloaded, informing the necessary schedule for a preventative upgrade.<br>
<br>
This machine learning pipeline uses:
1.	Python for the scripting.
2.	Docker containers to encapsulate training, promotion, and prediction jobs.
3.	MLflow for model management.
4.	YAML for job management.
5.	GitHub Actions to trigger a pipeline deployment.
6.	Terraform to create and update AzureML infrastructure from code.
7.	A managed identity to keep everything secure.
 
#1 through #3 allow for the entire pipeline to be run locally for quick feedback on changes before deploying to the cloud. No Azure required.
#6 allows for other cloud providers, such as AWS or GCP, to be swapped in as needed.

**Caveat**:
I only spent a few days on this so, though it is a working pipeline and ran multiple jobs successfully (screenshots below), it is of course bare bones and just a proof-of-concept. There are a bunch of features in AzureML and MLflow that I didn’t make use of due to lack of time. Additionally, it’s not the most interesting data science problem since it has only one feature feeding into the predictions. But it serves the larger purpose of illustrating how to setup machine learning pipeline infrastructure.
 If I had a few weeks or months, I could create a rich, production quality pipeline. But this pipeline is at least production-*__ready__* in terms of the way the system is designed and the technologies used. And since it addresses a hypothetical problem for the purposes of experimentation rather than an actual problem that a business is trying to solve, I’ll be spending limited time on it.

## AzureML In Action ##
This is the starting point: a high-level view of the experiment in Azure AI’s Machine Learning Studio.
<br><br> 
<img width="975" height="334" alt="image" src="https://github.com/user-attachments/assets/2e3193c1-84c8-4937-86bd-25a5e98250c5" />
<br><br>
Drilling into the experiment shows a list of its jobs. Each job represents a different deployment of the pipeline that trains the model, promotes the model (if it passed testing), and uses the model to make predictions.
<br><br>
<img width="975" height="361" alt="image" src="https://github.com/user-attachments/assets/90b3d854-d502-4018-b1f4-be952297215e" />
<br><br>
Drilling into the latest job reveals a list of sub-jobs and how they are wired together. Below you can see that the sub-job which trains the model outputs a “trained_model” to the job that promotes the model, which then outputs a “promoted_model” to the job that uses the model to output “predictions”.
<br><br>
<img width="975" height="525" alt="image" src="https://github.com/user-attachments/assets/6195e1ef-17ec-4b77-a483-8fb3ff01b42e" />
<br><br>
You can drill into each sub-job to view all kinds of details about it. Below you can see that the first sub-job, “Train Transformer Load Model”, did the following:
1.	It output the model; once when MLflow logged the model and once to pass the model along to the promotion job.
2.	It applied some informative tags.
3.	It reported the metric “rmse” (aka Root Mean Squared Error), indicating how well the model performed during testing.
<br><br> 
<img width="975" height="764" alt="image" src="https://github.com/user-attachments/assets/9d06b1c2-bdd0-4063-b8d9-c40e31041728" />
<br><br>
You can drill into one of the model links to get more information on the model.
<br><br>
<img width="975" height="568" alt="image" src="https://github.com/user-attachments/assets/58acce0a-c708-4e81-83d8-409edbb21873" />
<br><br>
And drill into its artifacts.
<br><br>
<img width="975" height="584" alt="image" src="https://github.com/user-attachments/assets/4c467143-1b8a-4d4e-b50f-338a9da7b92f" />
<br><br>
Moving on to the “Promote Transformer Load Model” sub-job, you can see that it output the “promoted_model”, meaning the model passed testing during training and the Root Mean Squared Error of the model was sufficiently low for it to be useful in making predictions.
<br><br>
<img width="975" height="533" alt="image" src="https://github.com/user-attachments/assets/39a7b34f-da4d-4dc8-b8cb-5af82bcbbcd7" />
<br><br>
If you view the AzureML model registry for the workspace, you can see that the promotion sub-job registered the model since it passed testing. 
<br><br>
<img width="975" height="390" alt="image" src="https://github.com/user-attachments/assets/9efb892f-2414-4b29-890a-35e52e1a1334" />
<br><br>
Moving onto the “Predict Transformer Overload” sub-job, we can see that it created the following:
1.	A tag reporting that the transformer is predicted to hit its first overload at 11pm on November 26th, 2027.
2.	A metric predicting that the maximum load over the entire 5-year period will be about 98 kilowatts.
3.	A metric predicting that the transformer will overload over 4,623 times in the next 5 years given current usage trends.
<br><br> 
<img width="975" height="766" alt="image" src="https://github.com/user-attachments/assets/90db621d-be48-42bd-ab21-55d48213aa86" />
<br><br>
You can also drill into the “predictions” output and see the files that the prediction job uploaded, including:
1.	The predicted transformer load in kilowatts for each hour during the next 5 years.
2.	The number of times the transformer is predicted to overload during that period.
3.	A chart of the predicted loads.
<br><br> 
<img width="975" height="648" alt="image" src="https://github.com/user-attachments/assets/123197c7-c266-4925-88e5-44d574411f0c" />

