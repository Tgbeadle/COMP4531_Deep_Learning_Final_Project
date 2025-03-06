# COMP4531_Deep_Learning_Final_Project

# Deep Learning Traffic Safety Analysis
This repository contains the final project for COMP 4531 - Deep Learning, utilizing TensorFlow and Jupyter to analyze traffic safety at intersections through neural networks.

## Problem Statement

Can a neural network accurately identify dangerous intersections from satellite imagery and crash data?

Traffic fatalities remain a critical public safety concern in the United States, with over 42,000 deaths recorded in 2022 ([US DOT Highway Safety](https://highways.dot.gov/safety/about-safety)). While not all of these deaths occurred at intersections, these junctions represent concentrated points of risk within the road network. Traditional traffic safety analysis relies on historical crash data and manual assessment, but modern machine learning techniques may offer new insights into identifying dangerous intersection designs before accidents occur.

A neural network may be able to predict dangerous intersections based on satellite imagery by identifying commonalities across intersections with high rates of accidents. It is important to note that there are limitations with using only satellite imagery as these images do not reflect weather conditions, time of day, past construction, impaired or distracted drivers, or other temporary hazards. This project examines whether the most dangerous intersections share visual features that a Convolutional Neural Network (CNN) can learn to identify. Some hazards may be obvious, like buildings creating blind corners, but the CNN could also identify less apparent patterns useful in predicting high risk intersections.

## Data Collection and Preprocessing Methodology

Accident data came from Denver Open Data Catalog. The set consists of "accident data from the previous five calendar years plus current year to date." [https://arcg.is/0zHWPm](https://arcg.is/0zHWPm)

Dataset includes 257,299 police traffic accident records from January 2013 through October 2024. Analysis focused on precincts 113 and 111, which is bounded by:
- East/West:
    - Pecos St.
    - Sheridan Rd.
- North/South: 
    - 50th St.
    - 29th St

For the purpose of this project, an intersection is defined as any point at which a road, ramp, or other driving surface merges or crosses another. Only accidents marked as 'Intersection Related', 'At Intersection', 'ROUNDABOUT', 'Ramp Related', or 'Ramp' in the accident data set were considered to exclude highway or other incidents that were within 100 feet of intersections but not related to the nearby intersection.

452 intersections were captured at a zoom of 20, then cropped to 1000x1000 pixels before being reduced to 255 x 255 for training.

After intersection points were identified and screenshots collected, crash locations were compared to all chosen intersections. If a crash was within 100 feet of an intersection (typical city intersection being 50-150 feet), 1 was added to the 'crash_count' column of the dataframe. If a crash was within 100 feet of two or more intersections, the crash was assigned to the closest intersection. In total, 3,523 accidents were attributed to the 452 intersections analyzed.

*All of the above was completed in a separate Jupyter notebook due to TensorFlow's "quirks" with other packages.*

## Docker Environment

This project uses the TensorFlow Jupyter Docker image `tensorflow/tensorflow:latest-jupyter` to provide a consistent development environment.

### Docker Image Details

- **Base image:** ubuntu:22.04
- **TensorFlow version:** 2.17.0 (CPU version)
- **Python version:** 3.11
- **Working directory:** /tf
- **Exposed ports:** 8888/tcp (Jupyter notebook)

### Running the Project

```bash
# Pull the image
docker pull tensorflow/tensorflow:latest-jupyter

# Run the container
docker run -p 8888:8888 -v $(pwd):/tf/project tensorflow/tensorflow:latest-jupyter
```

After starting the container, access the Jupyter notebook by opening http://localhost:8888 in your browser. You may need to copy the token that appears in the console output.

## Project Structure

- `/tf/project/notebooks/` - Jupyter notebooks for data analysis and model training
- `/tf/project/data/` - Processed intersection data and images
- `/tf/project/models/` - Saved model checkpoints

## Requirements

This project uses the following key libraries:
- TensorFlow 2.17.0
- Jupyter Notebook
- NumPy
- Matplotlib
- Pandas

All dependencies are included in the Docker image.
