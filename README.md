# Colorimetry project - IPSI3

## Description

This project is about finding a way to print a certain pattern on a paper sheet that would appear under a specific light source. The goal is to find the right colors to print on the paper so that the pattern is invisible under a classic white light, but appears under a specific light source.

## Authors

- [**Antoine Duteyrat**](https://github.com/antoinedenovembre)
- [**Kyllian Mosnat**](https://github.com/kmosnat)

## Installation

To be able to run the project, you need to have Python 3 installed on your computer. You can then download the prerequisites by running the following command:

```bash
pip install -r requirements.txt
``` 

## To-do list

- [x] Compute the Lab values for each patch under each illuminant
- [x] Compute the Delta E 00 between each patch under each illuminant
- [x] Find the most optimized couple of patches (maximizing delta E 00 under one illuminant and minimizing under another)
- [ ] Reinterpret the results with the printer's RGB curves to get the right colors to print for the patches
