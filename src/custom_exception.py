import sys


class CustomException(Exception):
    def __init__(self, message, error_detail):
        super().__init__(message)
        self.error_detail = error_detail

    def __str__(self):
        return f"{self.args[0]} | Error Details: {self.error_detail}"