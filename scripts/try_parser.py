"""
Test stage 1: turn a project idea into a ProjectSpec.

    python -m scripts.try_parser "predict whether a telecom customer will churn"
"""
import sys

from dataprep.parser import parse_requirements, scope_problem


def main():
    idea = " ".join(sys.argv[1:]).strip() or input("Project idea: ")
    spec = parse_requirements(idea)
    print(spec.model_dump_json(indent=2))
    problem = scope_problem(spec)
    if problem:
        print(f"\nNOTE: {problem}")


if __name__ == "__main__":
    main()
