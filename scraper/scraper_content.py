import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# Base URL of Python documentation
BASE_URL = "https://docs.python.org/3/"

# Function to scrape content from a single page
def scrape_page(url):
    """Scrapes text content from a given documentation page."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract main content area
        main_content = soup.find('div', {'class': 'body'})
        if not main_content:
            print(f"No content found on {url}")
            return ""

        # Extract and join all paragraphs as text
        return "\n".join([p.get_text(strip=True) for p in main_content.find_all('p')])
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return ""

# Function to find all section links on a page
def get_all_links(page_url):
    """Extracts all internal documentation links from a given page."""
    try:
        response = requests.get(page_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        links = []
        toc_div = soup.find('div', {'class': 'toctree-wrapper compound'})
        if toc_div:
            for a_tag in toc_div.find_all('a', href=True):
                href = a_tag["href"]
                if not href.startswith("#"):  # Ignore anchor links
                    links.append(urljoin(page_url, href))
        return links
    except Exception as e:
        print(f"Error retrieving links from {page_url}: {e}")
        return []

# Function to scrape a section and all its linked pages
def scrape_section(section_url):
    """Scrapes a section page and all links found within it."""
    print(f"📄 Scraping section: {section_url}")
    section_text = scrape_page(section_url)  # Get main section text
    all_links = get_all_links(section_url)  # Get all links from TOC

    # Scrape all linked pages and append content
    for link in all_links:
        print(f"🔗 Scraping sub-page: {link}")
        section_text += "\n\n" + scrape_page(link)

    return section_text

# Function to scrape all sections from the main documentation table
def scrape_python_docs():
    """Scrapes the entire Python documentation, including all linked sections."""
    try:
        response = requests.get(BASE_URL)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # Locate the table containing main sections
        table = soup.find('table', {'class': 'contentstable'})
        if not table:
            print("No table found.")
            return ""

        all_text = ""
        for a_tag in table.find_all('a', {'class': 'biglink'}, href=True):
            section_url = urljoin(BASE_URL, a_tag['href'])

            # Skip 'What's New' section
            if "What's new" in a_tag.get_text():
                continue

            # Scrape section + all linked pages
            all_text += "\n\n" + scrape_section(section_url)

        return all_text
    except Exception as e:
        print(f"Error scraping Python docs: {e}")
        return ""

# Main execution block
if __name__ == "__main__":
    print("🚀 Scraping Python Docs (Including All Nested Links)...")
    docs_text = scrape_python_docs()

    # Save output to a text file
    with open("python_docs.txt", "w", encoding="utf-8") as f:
        f.write(docs_text)

    print("✅ Scraping completed! Data saved in python_docs.txt.")
