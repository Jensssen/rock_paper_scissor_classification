# This script is used to zip the complete model into one folder.
# It is then uploaded to AWS as a zip file because of sagemaker.

import tarfile

with tarfile.open('model.tar.gz', mode='w:gz') as archive:
	archive.add('mobilenet', recursive=True)
