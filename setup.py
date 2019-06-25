import setuptools

with open("README.md", "r") as fh:
	long_description = fh.read()

setuptools.setup(
	name="emocon",
	version="0.0.1",
	author="Ravi Shankar",
	author_email="rshanka3@jhu.edu",
	description="A package that lets you convert emotion in speech.",
	long_description=long_description,
	url="",
	packages=setuptools.find_packages(),
	classifiers=[
		"Programming Language :: Python :: 3", 
		"Licence :: OSI Approved :: MIT Licence",
		"Operating System :: OS Independent",
		],
	)