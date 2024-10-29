# ElEx: Electrical Experiment Automation Suite

ElEx is a Python-based software suite designed for controlling multiple electrical measurement instruments, primarily from Keithley, for conducting automated electrical experiments. The suite aims to provide a uniform interface for reading and stimulating voltage and current across various instruments, enabling streamlined experiment design and execution.

## Features

- **Instrument Control**: Control multiple Keithley instruments for voltage/current measurements and stimulation.
- **Experiment Scripting**: Uniform scripts for setting up and executing experiments.
- **Data Handling**: Collect, log, and analyze measurements with ease.

## Requirements

The required Python packages are listed in `requirements.txt`. You can install them using:

```bash
pip install -r requirements.txt
```

## Getting Started

1. Clone the repository:

   ```bash
   git clone https://github.com/nanotechdave/ElEx.git
   cd ElEx
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the main script:

   ```bash
   python main.py
   ```

   This will execute the main experiment control flow.

## Usage

- Modify the `config.json` file to set up your experiment parameters.
- Use `main.py` to execute experiments based on the configured setup.

## Structure

- `main.py`: Entry point for executing the main experiment flow.
- `instruments/`: Contains scripts for interfacing with specific instruments.
- `utils/`: Utility scripts for data handling, logging, and analysis.

## Contributing

Contributions are welcome! Feel free to fork the repository and submit pull requests.

## License

This project is licensed under the MIT License. See `LICENSE` for more information.

## Contact

For any questions or suggestions, please open an issue or contact the repository owner.

