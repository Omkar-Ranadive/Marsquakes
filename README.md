# Marsquakes 
This repository can be used to analyze seismic events on Mars. Following analyses can be performed - 
* Generating rayplots 
* Calculating seismic event distances, generating distance plots 
* Calculating back-azimuth, azimuth, generating energy plots 
* Generating frequency filter plots 
* Fault guessing 
* Generating beach ball plots 

## Installation 
* Download [Miniconda](https://docs.conda.io/en/latest/miniconda.html) for your OS 
* Create a new conda environment: 
```
conda create -n mars python=3.10
```
* Activate the new conda environment: 
```
conda activate mars
```
* Clone the Marsquake repository: 
```
git clone https://github.com/Omkar-Ranadive/Marsquakes.git
```
* Go to project directory 
```
cd Marsquakes
```
* Install required libraries 
```
pip install -r requirements.txt
```

## Project Structure
* data/models - Place your interior structure models (.nd format) inside the data/models folder. The repository already comes with 4 different interior structure models - Gudkova, NewGudkova, TAYAK, Combined. 
* experiments/ - Experiment directories will be created under this folder whenever any piece of code is executed. 
* src/ - The actual executable files are placed inside this folder 


## Execution 
### Notes
* All executable files are placed in src directory. Go to the src directory to run the files: 
```
cd src
```
* All executable files have customizable parameters. To get a list of what the parameters do, the following command can be executed (replace file_name.py with file of interest): 
```
python file_name.py --help
```
* **Whenever a file is executed, relevant information related to the execution gets stored in experiments/exp_dir/info.log file.** 

### 1. Building models and generating rayplots 
* model_build.py - This file can be used to convert the models in .nd format to Taup Model format which is necessary for Obspy. Additionally, ray plots can be generated for each converted model. 
* Before running model build, make sure that all .nd format models are placed in data/models folder. 
* Example execution - To convert NewGudkova and TAYAK to TauP format and generate rayplots with default parameters for these models and place them in a exp dir called "results", run the following command: 
```
python model_build.py --exp_dir results -m NewGudkova TAYAK --rayplot
```
* Converted models are stored in experiments/exp_dir/models and the rayplots are stored in experiments/exp_dir/plots/rayplots
* Additional info on the execution can be found in experiments/exp_dir/info.log file.

### 2. Calculating event distances and generating distance plots 
* dist_cal.py - This file can be used to generate event dsitances and distance plots. 
* A file containing information about event start and end times is required. Store such a file in **json format** under experiments/exp_dir/events folder. 
* For the purpose of this example, we will be using a file named events_info.txt placed in experiments/results/events with the following content: 
```
{
"S0235b":
	{	
		"start": "2019-07-26T12:19:18.70", 
		"end": "2019-07-26T12:22:05.7", 
		"baz": 74
	},

"S0173a":
	{	
		"start": "2019-05-23T02:22:59.48", 
		"end": "2019-05-23T02:25:53.27", 
		"baz": 91
	},
	
"S0173ab":
	{	
		"start": "2019-05-23T02:23:03.30", 
		"end": "2019-05-23T02:25:56.9", 
		"baz": 80
	},

"S0325a":
	{	
		"start": "2019-10-26T06:58:58.0", 
		"end": "2019-10-26T07:02:49.3", 
		"baz": 123
	},

"S0325ab":
	{	
		"start": "2019-10-26T06:59:08.23", 
		"end": "2019-10-26T07:02:59.9" 
	}
}
```
* The **start** and **end** keys denote the p-arrival time and s-arrival time for the events. The baz key is *optional* and only used for comparison purposes later on. 
* Example execution - To calculate distances for these events with NewGudkova as interior structure model and source depth of 35km, run the following command: 
```
python dist_cal.py --exp_dir results --file events_info.txt --model NewGudkova --depth 35
```
* The distances get stored in a dictionary under experiments/exp_dir/events and are also written in info.log file. 
* Distance plots can be found in experiments/exp_dir/plots/distance_plots
* Additionally, distances across different models and depths can be compared using ddcompare.py as follows: 
```
python ddcompare.py --exp_dir results --file events_info.txt --depths 15 25 35 45 50 55
```

### 3. Calculating back-azimuth and azimuth 
* az_cal.py - This file can be used to calculate the back-azimuth, azimuth, and lat/lon of the events. 
* Similar to dist_cal.py, a text file in json format is required which provides the start (p-arrival) and end (s-arrival) times of the events. 
* Example execution - To calculate using events_info.txt and using the distances calculated from New Gudkova, 35km depth model, run the following command: 
```
python az_cal.py --exp_dir results --file events_info.txt --model NewGudkova --depth 35
```
* A dictionary containing the dist, baz, az, lat and lon gets saved under experiments/exp_dir/events folder and the information is also written in info.log file. 
* P wave and S wave plots for different events are saved in the experiments/exp_dir/plots/wave_plots folder. 

### 4. Fault Guessing 
* fault_guess.py - This file can be used for calculating the faulting mechanism. 
* A text file in json format containing the observed P, SH, SV and the error P, SH, SV is required. For this example, we will use a events_amps.txt placed in experiments/exp_dir/events folder with the following content: 
```
{
"S0235b":
	{	
		"obsP": 3.62e-10,
		"obsSV": 3.47e-9, 
		"obsSH": -1.611e-9, 
		"errP": 3.73e-11, 
		"errSV": 7.03e-11, 
		"errSH": 7.42e-11 
	},

"S0173a":
	{	
		"obsP": -1.25e-9,
		"obsSV": 0.955e-9, 
		"obsSH": -0.371e-9, 
		"errP": 1.13e-10, 
		"errSV": 1.46e-10, 
		"errSH": 2.01e-10 
	},
	
"S0173ab":
	{	
		"obsP": 1.09e-9,
		"obsSV": -1.22e-9, 
		"obsSH": -3.29e-9, 
		"errP": 1.13e-10, 
		"errSV": 1.46e-10, 
		"errSH": 2.01e-10 
	},

"S0325a":
	{	
		"obsP": 5.38e-10,
		"obsSV": -1.44e-9, 
		"obsSH": -9.63e-10, 
		"errP": 1.75e-10, 
		"errSV": 4.79e-10, 
		"errSH": 4.25e-10 
	},

"S0325ab":
	{	
		"obsP": -1.35e-9,
		"obsSV": -4.06e-9, 
		"obsSH": 0.25e-9, 
		"errP": 1.75e-10, 
		"errSV": 4.79e-10, 
		"errSH": 4.25e-10 
	}
}
```
* Example execution - To get the faulting mechanism for these events with NewGudkova model, 35km depth, run the following command: 
```
python fault_guess.py --exp_dir results --file events_amps.txt --model NewGudkova --depth 35 
```
* .csv file containing the misfit value for different strike, dip, range combinations is saved under experiments/exp_dir/events/event_name for each event. 

### 5. Beach ball plots 
* beach_plot.py - The .csv files generated using fault_guess.py can be used to create beach ball plots of the solution set. 
* Example execution - 
```
python beach_plot.py --exp_dir results --model NewGudkova --depth 35
```
* The beach ball plots are saved under experiments/exp_dir/plots/beach_plots directory. 

### 6. Frequency filter plots 
* Events can be filtered across different frequency bands as follows: 
```
python filter_plots.py --exp_dir results --file events_info.txt --scales 80000 5000 80000 80000 5000 2000 1000
```
## References 
This repository is an updated and modular version of [Sita Marsquakes 2020](https://github.com/maddysita-17/Sita_Marsquakes2020) repository. 
