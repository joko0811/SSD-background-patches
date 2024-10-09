import git


def get_git_sha():
    try:
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        return sha
    except Exception as e:
        return "In this environment it was not possible to get the git commit hash."

