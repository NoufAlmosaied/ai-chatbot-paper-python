"""
HTML Fetcher Service
Safely fetches and parses HTML content from URLs
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import logging
from typing import Optional, Dict
import time

logger = logging.getLogger(__name__)


class HTMLFetcher:
    """Safely fetch and parse HTML content from URLs."""

    def __init__(self, timeout: int = 5, max_size: int = 5 * 1024 * 1024):
        """
        Initialize HTML fetcher.

        Args:
            timeout: Request timeout in seconds (default: 5)
            max_size: Maximum page size in bytes (default: 5MB)
        """
        self.timeout = timeout
        self.max_size = max_size
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })

    def fetch(self, url: str) -> Optional[Dict]:
        """
        Fetch and parse HTML content from URL.

        Args:
            url: URL to fetch

        Returns:
            Dictionary with 'html', 'soup', 'url', 'final_url' or None if fetch fails
        """
        try:
            # Validate URL format
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                logger.warning(f"Invalid URL format: {url}")
                return None

            # Fetch HTML with timeout
            start_time = time.time()
            response = self.session.get(
                url,
                timeout=self.timeout,
                allow_redirects=True,
                stream=True,
                verify=True  # Verify SSL certificates
            )

            # Check response status
            if response.status_code != 200:
                logger.warning(f"HTTP {response.status_code} for {url}")
                return None

            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            if 'text/html' not in content_type:
                logger.warning(f"Non-HTML content type: {content_type} for {url}")
                return None

            # Read content with size limit
            content = b''
            for chunk in response.iter_content(chunk_size=8192):
                content += chunk
                if len(content) > self.max_size:
                    logger.warning(f"Page too large (>{self.max_size} bytes): {url}")
                    return None

            # Decode content
            html = content.decode(response.encoding or 'utf-8', errors='ignore')

            # Parse HTML
            soup = BeautifulSoup(html, 'lxml')

            fetch_time = time.time() - start_time
            logger.info(f"Fetched {url} in {fetch_time:.2f}s ({len(html)} bytes)")

            return {
                'html': html,
                'soup': soup,
                'url': url,
                'final_url': response.url,  # After redirects
                'status_code': response.status_code,
                'headers': dict(response.headers),
                'fetch_time': fetch_time
            }

        except requests.exceptions.Timeout:
            logger.warning(f"Timeout fetching {url}")
            return None
        except requests.exceptions.SSLError:
            logger.warning(f"SSL error fetching {url}")
            return None
        except requests.exceptions.ConnectionError:
            logger.warning(f"Connection error fetching {url}")
            return None
        except requests.exceptions.TooManyRedirects:
            logger.warning(f"Too many redirects for {url}")
            return None
        except Exception as e:
            logger.error(f"Error fetching {url}: {str(e)}")
            return None

    def extract_links(self, soup: BeautifulSoup, base_url: str) -> Dict:
        """
        Extract all hyperlinks from HTML.

        Args:
            soup: BeautifulSoup object
            base_url: Base URL for resolving relative links

        Returns:
            Dictionary with link analysis
        """
        all_links = soup.find_all('a', href=True)
        base_domain = urlparse(base_url).netloc

        internal_links = []
        external_links = []
        null_links = []

        for link in all_links:
            href = link.get('href', '').strip()

            # Handle null/self-redirect links
            if href in ['#', '', 'javascript:void(0)', 'javascript:;']:
                null_links.append(href)
                continue

            # Resolve relative URLs
            absolute_url = urljoin(base_url, href)
            link_domain = urlparse(absolute_url).netloc

            # Classify as internal or external
            if link_domain == base_domain or not link_domain:
                internal_links.append(absolute_url)
            else:
                external_links.append(absolute_url)

        total_links = len(all_links)

        return {
            'total': total_links,
            'internal': len(internal_links),
            'external': len(external_links),
            'null': len(null_links),
            'pct_external': len(external_links) / total_links if total_links > 0 else 0.0,
            'pct_null': len(null_links) / total_links if total_links > 0 else 0.0,
            'internal_list': internal_links,
            'external_list': external_links
        }

    def extract_resources(self, soup: BeautifulSoup, base_url: str) -> Dict:
        """
        Extract all resource URLs (images, scripts, stylesheets).

        Args:
            soup: BeautifulSoup object
            base_url: Base URL for resolving relative URLs

        Returns:
            Dictionary with resource analysis
        """
        base_domain = urlparse(base_url).netloc

        resources = {
            'images': soup.find_all('img', src=True),
            'scripts': soup.find_all('script', src=True),
            'stylesheets': soup.find_all('link', href=True, rel='stylesheet')
        }

        internal_resources = 0
        external_resources = 0
        total_resources = 0

        for resource_type, elements in resources.items():
            for element in elements:
                src = element.get('src') or element.get('href', '')
                if not src:
                    continue

                total_resources += 1
                absolute_url = urljoin(base_url, src)
                resource_domain = urlparse(absolute_url).netloc

                if resource_domain == base_domain or not resource_domain:
                    internal_resources += 1
                else:
                    external_resources += 1

        return {
            'total': total_resources,
            'internal': internal_resources,
            'external': external_resources,
            'pct_external': external_resources / total_resources if total_resources > 0 else 0.0
        }

    def analyze_forms(self, soup: BeautifulSoup, base_url: str) -> Dict:
        """
        Analyze forms on the page.

        Args:
            soup: BeautifulSoup object
            base_url: Base URL

        Returns:
            Dictionary with form analysis
        """
        forms = soup.find_all('form')
        base_domain = urlparse(base_url).netloc

        insecure_forms = 0
        external_forms = 0
        relative_forms = 0
        abnormal_forms = 0
        submit_to_email = 0

        for form in forms:
            action = form.get('action', '').strip()
            method = form.get('method', 'get').lower()

            # Check for empty or relative action
            if not action or action in ['', '#']:
                relative_forms += 1
                continue

            # Check for email submission
            if action.startswith('mailto:'):
                submit_to_email += 1
                abnormal_forms += 1
                continue

            # Resolve absolute URL
            absolute_action = urljoin(base_url, action)
            action_parsed = urlparse(absolute_action)

            # Check if insecure (HTTP not HTTPS)
            if action_parsed.scheme == 'http':
                insecure_forms += 1

            # Check if external domain
            if action_parsed.netloc and action_parsed.netloc != base_domain:
                external_forms += 1
                abnormal_forms += 1

        total_forms = len(forms)

        return {
            'total': total_forms,
            'insecure': insecure_forms,
            'external': external_forms,
            'relative': relative_forms,
            'abnormal': abnormal_forms,
            'submit_to_email': submit_to_email,
            'has_insecure': 1 if insecure_forms > 0 else 0,
            'has_external': 1 if external_forms > 0 else 0,
            'has_relative': 1 if relative_forms > 0 else 0,
            'has_abnormal': 1 if abnormal_forms > 0 else 0
        }

    def check_favicon(self, soup: BeautifulSoup, base_url: str) -> int:
        """
        Check if favicon is external.

        Args:
            soup: BeautifulSoup object
            base_url: Base URL

        Returns:
            1 if favicon is external, 0 otherwise
        """
        base_domain = urlparse(base_url).netloc

        # Find favicon link
        favicon = soup.find('link', rel=lambda x: x and 'icon' in x.lower())
        if not favicon:
            return 0

        href = favicon.get('href', '')
        if not href:
            return 0

        # Resolve absolute URL
        absolute_url = urljoin(base_url, href)
        favicon_domain = urlparse(absolute_url).netloc

        # Check if external
        return 1 if favicon_domain and favicon_domain != base_domain else 0

    def check_page_features(self, soup: BeautifulSoup, html: str) -> Dict:
        """
        Check various page-level features.

        Args:
            soup: BeautifulSoup object
            html: Raw HTML string

        Returns:
            Dictionary with page features
        """
        features = {}

        # Check for iframes
        features['has_iframe'] = 1 if soup.find('iframe') or soup.find('frame') else 0

        # Check for missing title
        title = soup.find('title')
        features['missing_title'] = 1 if not title or not title.get_text().strip() else 0

        # Check for right-click disable (JavaScript)
        html_lower = html.lower()
        features['right_click_disabled'] = 1 if any(x in html_lower for x in [
            'oncontextmenu="return false"',
            'event.button==2',
            'preventdefault',
            'disableselect'
        ]) else 0

        # Check for popup windows (JavaScript)
        features['has_popup'] = 1 if 'window.open' in html_lower else 0

        # Check for status bar manipulation (JavaScript)
        features['fake_status_bar'] = 1 if any(x in html_lower for x in [
            'window.status',
            'onmouseover="window.status'
        ]) else 0

        # Check for images-only forms (no text input)
        forms = soup.find_all('form')
        images_only_forms = 0
        for form in forms:
            inputs = form.find_all(['input', 'textarea', 'select'])
            images = form.find_all('img')
            if len(images) > 0 and len(inputs) == 0:
                images_only_forms += 1
        features['images_only_in_form'] = 1 if images_only_forms > 0 else 0

        return features

    def analyze_domain_mismatch(self, soup: BeautifulSoup, base_url: str) -> int:
        """
        Check for frequent domain name mismatches in links.

        Args:
            soup: BeautifulSoup object
            base_url: Base URL

        Returns:
            1 if frequent mismatches detected, 0 otherwise
        """
        links = soup.find_all('a', href=True)
        base_domain = urlparse(base_url).netloc

        if len(links) == 0:
            return 0

        mismatches = 0
        for link in links:
            href = link.get('href', '')
            if href.startswith('http'):
                link_domain = urlparse(href).netloc
                if link_domain and link_domain != base_domain:
                    mismatches += 1

        mismatch_ratio = mismatches / len(links)

        # Frequent mismatch if more than 60% of links go to different domains
        return 1 if mismatch_ratio > 0.6 else 0
