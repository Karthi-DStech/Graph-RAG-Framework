import argparse
import os
import sys


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class BaseOptions:
    """
    This class defines the base options for the script.

    Parameters
    ----------
    None

    Methods
    -------
    initialize()
        Initializes the parser with the required arguments.

    parse()
        Parses the arguments passed to the script.

    Returns
    -------
    opt: argparse.Namespace
        The parsed arguments.
    """

    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self) -> None:
        self.parser.add_argument(
            "--LangChainAPIKey",
            type=str,
            default="<yoour-api-key>",
            help="API key for LangChain",
        )

        self.parser.add_argument(
            "--OpenAPIKey",
            type=str,
            default="<your-api-key>",
            help="API key for OpenAI",
        )

        self.parser.add_argument(
            "--Neo4jURI",
            type=str,
            default="neo4j+s://your-neo4j-instance",
            help="URI for Neo4j database collection",
        )

        self.parser.add_argument(
            "--Neo4jUserName",
            type=str,
            default="your-username",
            help="Username for Neo4j database collection",
        )

        self.parser.add_argument(
            "--Neo4jPassword",
            type=str,
            default="your-password",
            help="Password for Neo4j database collection",
        )

        self.parser.add_argument(
            "--Neo4jDatabase",
            type=str,
            default="your-database",
            help="Database name for Neo4j database collection",
        )

        self.initialized = True

    def parse(self):
        """
        Parses the arguments passed to the script

        Parameters
        ----------
        None

        Returns
        -------
        opt: argparse.Namespace
            The parsed arguments
        """
        if not self.initialized:
            self.initialize()
        self._opt = self.parser.parse_args()

        return self._opt
