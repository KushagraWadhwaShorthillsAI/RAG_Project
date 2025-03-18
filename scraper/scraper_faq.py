import requests
from bs4 import BeautifulSoup
import csv

BASE_URL = "https://docs.python.org/3/faq/"

def get_faq_links():
    """Fetch all FAQ category links from the main index page."""
    url = BASE_URL + "index.html"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    
    faq_links = []
    for li in soup.select("div.toctree-wrapper ul li a.reference.internal"):
        faq_links.append(BASE_URL + li["href"])  # Create absolute URL
    return faq_links

def scrape_faq_page(url):
    """Extract FAQ questions and answers from a given FAQ page."""
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    
    faqs = []
    for section in soup.find_all("section"):  # Each FAQ is in a <section>
        question_tag = section.find("h3")
        if question_tag:
            question = question_tag.get_text(strip=True)
            
            # Collect all <p> tags under this section (answers are in multiple <p>)
            answer_parts = [p.get_text(" ", strip=True) for p in section.find_all("p")]
            answer = " ".join(answer_parts)  # Join paragraphs into one answer
            
            faqs.append((question, answer))
    
    return faqs

def save_to_csv(faqs):
    """Save extracted FAQs to a CSV file using '|' as the separator."""
    with open("python_faqs.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, delimiter="|")
        writer.writerow(["Question", "Answer"])  # Header row
        writer.writerows(faqs)

def main():
    all_faqs = []
    faq_links = get_faq_links()
    
    for link in faq_links:
        print(f"Scraping: {link}")
        faqs = scrape_faq_page(link)
        all_faqs.extend(faqs)

    save_to_csv(all_faqs)
    print("Scraping complete! FAQs saved to python_faqs.csv")

if __name__ == "__main__":
    main()
