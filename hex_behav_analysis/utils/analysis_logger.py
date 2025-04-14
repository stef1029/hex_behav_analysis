# analysis_logger.py
from colorama import Fore, Back, Style
import inspect

class AnalysisLogger:
    """
    Custom logger for the Analysis Manager that supports colorized output
    and verbose control.
    """
    def __init__(self):
        # Dictionary mapping function names to colors
        self.function_colors = {
            # Default is cyan - used when function is not in this list
            "default": Fore.CYAN,
            # Add specific function colors here
            "load_ephys_timestamps": Fore.MAGENTA,
            "get_camera_frame_times": Fore.YELLOW,
            "get_scales_data": Fore.GREEN,
            "ingest_behaviour_data": Fore.BLUE,
            "clean_DAQ_data": Fore.RED,
            "pulse_ID_sync": Fore.LIGHTBLUE_EX,
            # Add more functions and colors as needed
        }
        
        # Set this to control verbosity globally
        self.verbose = True
        
        # Main tag color
        self.main_tag_color = Fore.CYAN
        
    def log(self, func_name, message, force=False):
        """
        Log a message with appropriate coloring based on the function name.
        
        Args:
            func_name (str): Name of the calling function
            message (str): Message to log
            force (bool): If True, print regardless of verbose setting
        """
        if not self.verbose and not force:
            return
        
        # Get the color for this function
        func_color = self.function_colors.get(func_name, self.function_colors["default"])
        
        # Format and print the message
        if func_name == "default":
            # Don't print the function name for "default"
            print(f"{self.main_tag_color}Analysis Manager:{Style.RESET_ALL} {message}")
        else:
            # Print function name for specific functions
            print(f"{self.main_tag_color}Analysis Manager:{Style.RESET_ALL} {func_color}{func_name}:{Style.RESET_ALL} {message}")
    
    def error(self, func_name, message):
        """
        Log an error message - always prints regardless of verbose setting
        
        Args:
            func_name (str): Name of the calling function
            message (str): Error message to log
        """
        if func_name == "default":
            self.log(func_name, f"{Fore.RED}{message}{Style.RESET_ALL}", force=True)
        else:
            self.log(func_name, f"{Fore.RED}{message}{Style.RESET_ALL}", force=True)

    
    def set_verbose(self, verbose):
        """
        Set the verbose flag
        
        Args:
            verbose (bool): Whether to print verbose messages
        """
        self.verbose = verbose
    
    def add_function_color(self, func_name, color):
        """
        Add or update a function color in the mapping
        
        Args:
            func_name (str): Function name
            color: Colorama Fore color to use
        """
        self.function_colors[func_name] = color