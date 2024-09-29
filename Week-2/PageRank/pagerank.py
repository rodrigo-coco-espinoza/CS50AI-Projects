import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """

    # If the page has no outgoing links
    if not corpus[page]:
        # Calculate the probability of choosing a link at random
        probability_random = 1 / len(corpus)

        # Initialize the probability distribution
        probability_distribution = {key: probability_random for key in corpus.keys()}

        return probability_distribution

    # Calculate the probability of choosing a link at random
    probability_random = (1 - DAMPING) / len(corpus)

    # Initialize the probability distribution
    probability_distribution = {key: probability_random for key in corpus.keys()}

    # Calculate the probability of choosing a link from the current page
    for key in corpus[page]:
        probability_distribution[key] += DAMPING / len(corpus[page])

    return probability_distribution


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    # Initialize the dictionary to store the number of times each page is visited
    page_visits = {key: 0 for key in corpus.keys()}

    # Choose a page at random
    page = random.choice(list(corpus.keys()))

    # Iterate over the number of samples
    for _ in range(n):
        # Update the page visits
        page_visits[page] += 1

        # Calculate the probability distribution
        probability_distribution = transition_model(corpus, page, damping_factor)

        # Choose the next page based on the probability distribution
        page = random.choices(list(probability_distribution.keys()),
                              weights=list(probability_distribution.values()))[0]

    # Calculate the estimated PageRank for each page
    page_rank = {key: value / n for key, value in page_visits.items()}

    return page_rank


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    # Initialize the PageRank
    page_rank = {key: 1 / len(corpus) for key in corpus.keys()}

    convergence = False

    damping_probability = (1 - damping_factor) / len(corpus)

    while not convergence:
        new_page_rank = {key: 0 for key in corpus.keys()}

        for page in corpus.keys():
            sum_page_rank = 0

            for page_i in corpus.keys():
                if len(corpus[page_i]) == 0:
                    sum_page_rank += 1 / len(corpus)
                elif page in corpus[page_i]:
                    sum_page_rank += page_rank[page_i] / len(corpus[page_i])

            new_page_rank[page] = damping_probability + damping_factor * sum_page_rank

        # Normalize the PageRank
        new_page_rank = {key: value / sum(new_page_rank.values())
                         for key, value in new_page_rank.items()}

        # Check for convergence
        convergence = all(abs(new_page_rank[page] - page_rank[page])
                          < 0.001 for page in corpus.keys())

        # Update the PageRank
        page_rank = new_page_rank

    return page_rank


if __name__ == "__main__":
    main()
