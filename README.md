# Electric Jet Dynamic Requirements Analysis

## Overview

This project aims to analyze the dynamic requirements of an electric jet using a web application built with Streamlit. The application takes into account various parameters such as cruising speed, range, motor thrust, motor mass, and efficiencies to calculate key metrics. It compares these metrics with a well-known commercial jet, the Boeing 737-800 MAX, to provide a comprehensive analysis.

## Features

- User-friendly web interface for inputting system and subsystem level requirements.
- Calculation of target eTurbofan thrust and heat based on user inputs.
- Heat management mass calculation.
- Electric jet range calculation using a modified Breguet Range Equation tailored for electric aircraft.
- Metrics comparison with Boeing 737-800 MAX.

## Installation

1. Clone this repository:

```
git clone https://github.com/yourusername/electric-jet-dynamic-requirements.git
```

2. Navigate to the project directory:

```
cd electric-jet-dynamic-requirements
```

3. Install the required Python packages:

```
pip install -r requirements.txt
```

4. Run the Streamlit app:

```
streamlit run app.py
```

## Usage

1. Open the web application and navigate to the "System Level Requirements" section.
2. Input the target cruising speed, maximum range, and other requirements.
3. Navigate to the "Motor Subsystem" section.
4. Input the target eTurbofan thrust, maximum motor mass, and heat dissipation capacity.
5. Adjust the control points like efficiencies, lift-to-drag ratio, and battery specific energy.
6. Review the calculated metrics and their comparison with the Boeing 737-800 MAX.

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

