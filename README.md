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
I only spent a few days on this so, though it is a working pipeline and ran multiple jobs successfully (screenshots below), it is of course bare bones and just a proof-of-concept. There are a bunch of features in AzureML and MLflow that I didn’t make use of due to lack of time. Additionally, it’s not the most interesting data science problem since it has very few features that go into the predictions. But it serves the larger purpose of illustrating how to setup machine learning pipeline infrastructure.
 If I had a few weeks or months, I could create a rich, production quality pipeline. But this pipeline is at least production-ready in terms of the way the system is designed and the technologies used. And since it addresses a hypothetical problem for the purposes of experimentation rather than an actual problem that a business is trying to solve, I’ll be spending limited time on it.
