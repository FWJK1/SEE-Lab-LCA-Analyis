import git
from datetime import datetime
import time
from functools import wraps
import json

def get_git_root():
    repo = git.Repo(search_parent_directories=True)
    return repo.git.rev_parse("--show-toplevel")

def get_current_time():
    return datetime.now().strftime("%H:%M:%S")

def print_time():
    print(get_current_time)

def printline():
    print("---" * 50)

def log_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Record start time
        printline()
        print(f"Started {func.__name__} at {get_current_time()}")
        result = func(*args, **kwargs)
        end_time = time.time()  # Record end time
        elapsed_time = end_time - start_time  # Calculate elapsed time
        print(f"Finished {func.__name__} at {get_current_time()}")
        print(f"Total time elapsed: {elapsed_time:.4f} seconds")  # Print elapsed time
        printline()
        print("\n")
        return result
    return wrapper


def print_dict(dictionary):
    print(json.dumps(dictionary, indent=4))



    
# class SQLFilter(logging.Filter):
#     def __init__(self, exclude_patterns=None):
#         # Initialize with a list of patterns to exclude (default: empty list)
#         self.exclude_patterns = exclude_patterns or []

#     def filter(self, record):
#         # Get the message part of the log
#         message = record.getMessage()

#         # Check if the message (query) starts with any of the exclude_patterns
#         return not any(message.startswith(pattern) for pattern in self.exclude_patterns)

# # List of patterns to filter out
# exclude_patterns = ["SELECT", "INSERT", "UPDATE", "DELETE", "DROP"]

    
# logging.basicConfig(
#     level=print,  # Change to INFO, WARNING, etc. for different verbosity
#     format="%(message)s",
#     handlers=[
#         logging.StreamHandler(sys.stdout),  # Output to console (stdout)
#         logging.FileHandler("app.log")      # Output to a file
#     ],
# )
