import re
import time
from pathlib import Path
from urllib.parse import quote_plus

import requests
from selenium import webdriver
from selenium.common.exceptions import (
    StaleElementReferenceException,
    TimeoutException,
)
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


SCROLL_PAUSE_TIME = 1.0
PREVIEW_WAIT_TIME = 7
MAX_SCROLL_COUNT = 40


def sanitize_filename(text: str) -> str:
    """Remove characters that are not allowed in Windows filenames."""
    text = re.sub(r'[\\/:*?"<>|]', "_", text)
    text = text.strip(" .")

    return text or "images"


def get_thumbnail_elements(driver):
    """
    Find thumbnail image elements on the Google Images results page.

    Instead of relying on Google's temporary CSS class names, this function
    identifies possible thumbnails using their image URLs and displayed sizes.
    """
    elements = driver.find_elements(By.CSS_SELECTOR, "img")
    thumbnails = []

    for element in elements:
        try:
            src = (
                element.get_attribute("currentSrc")
                or element.get_attribute("src")
                or element.get_attribute("data-src")
                or ""
            )

            width = element.size.get("width", 0)
            height = element.size.get("height", 0)

            is_google_thumbnail = (
                "gstatic.com/images" in src
                or "encrypted-tbn" in src
                or src.startswith("data:image")
            )

            if (
                element.is_displayed()
                and width >= 100
                and height >= 70
                and is_google_thumbnail
            ):
                thumbnails.append(element)

        except StaleElementReferenceException:
            continue

    return thumbnails


def get_large_image_candidates(driver):
    """
    Return image URL candidates that appear to be original images
    or large preview images currently displayed on the page.
    """
    candidates = []

    for element in driver.find_elements(By.CSS_SELECTOR, "img"):
        try:
            if not element.is_displayed():
                continue

            src = (
                element.get_attribute("currentSrc")
                or element.get_attribute("src")
                or element.get_attribute("data-src")
                or ""
            )

            if not src.startswith(("http://", "https://")):
                continue

            # Exclude Google thumbnail images.
            if "encrypted-tbn" in src or "gstatic.com/images" in src:
                continue

            natural_width = driver.execute_script(
                "return arguments[0].naturalWidth || 0;",
                element,
            )

            natural_height = driver.execute_script(
                "return arguments[0].naturalHeight || 0;",
                element,
            )

            if natural_width >= 300 and natural_height >= 200:
                area = natural_width * natural_height
                candidates.append((area, src))

        except StaleElementReferenceException:
            continue

    # Place the largest images first.
    candidates.sort(reverse=True)

    return candidates


def wait_for_new_large_image(driver, previous_urls):
    """
    Wait until a new large preview image appears after clicking a thumbnail.
    """

    def find_url(current_driver):
        candidates = get_large_image_candidates(current_driver)

        for _, url in candidates:
            if url not in previous_urls:
                return url

        return False

    return WebDriverWait(
        driver,
        PREVIEW_WAIT_TIME,
    ).until(find_url)


def get_extension(content_type: str) -> str:
    """Determine the file extension based on the HTTP Content-Type."""
    content_type = content_type.lower().split(";")[0]

    extension_map = {
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
        "image/gif": ".gif",
        "image/bmp": ".bmp",
    }

    return extension_map.get(content_type, ".jpg")


def download_image(
    session,
    url,
    save_directory,
    keyword,
    number,
):
    """Download an image and save it to the specified directory."""
    response = session.get(
        url,
        timeout=15,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/140.0 Safari/537.36"
            ),
            "Referer": "https://www.google.com/",
        },
    )

    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "")

    if not content_type.lower().startswith("image/"):
        raise ValueError(
            "The response is not an image: "
            f"Content-Type={content_type}"
        )

    extension = get_extension(content_type)
    filename = f"{keyword}_{number:04d}{extension}"
    filepath = save_directory / filename

    filepath.write_bytes(response.content)

    return filepath


def main():
    search_word = input("Enter a search keyword: ").strip()

    requested_number = int(
        input("Enter the number of images to download: ").strip()
    )

    safe_keyword = sanitize_filename(search_word)

    # Set the directory where downloaded images will be stored.
    save_directory = (
        Path("C:/Users/build/AI_Pytorch/google_images")
        / safe_keyword
    )

    save_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    options = webdriver.ChromeOptions()

    options.add_experimental_option(
        "excludeSwitches",
        ["enable-logging"],
    )

    # Open Chrome in a maximized window.
    options.add_argument("--start-maximized")

    driver = webdriver.Chrome(options=options)

    session = requests.Session()
    downloaded_urls = set()

    try:
        search_url = (
            "https://www.google.com/search"
            f"?tbm=isch&hl=en&q={quote_plus(search_word)}"
        )

        driver.get(search_url)

        # Wait until image elements appear on the page.
        WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located(
                (By.CSS_SELECTOR, "img")
            )
        )

        # Scroll until enough thumbnail images have been loaded.
        for _ in range(MAX_SCROLL_COUNT):
            thumbnails = get_thumbnail_elements(driver)

            if len(thumbnails) >= requested_number * 2:
                break

            driver.execute_script(
                "window.scrollTo(0, document.body.scrollHeight);"
            )

            time.sleep(SCROLL_PAUSE_TIME)

            # Try to click the "Show more results" button if it exists.
            show_more_xpaths = [
                "//button[contains(., 'Show more results')]",
                "//button[contains(., '결과 더보기')]",
                "//input[contains(@value, 'Show more results')]",
                "//input[contains(@value, '결과 더보기')]",
            ]

            for xpath in show_more_xpaths:
                try:
                    buttons = driver.find_elements(
                        By.XPATH,
                        xpath,
                    )

                    if buttons and buttons[0].is_displayed():
                        driver.execute_script(
                            "arguments[0].click();",
                            buttons[0],
                        )

                        time.sleep(SCROLL_PAUSE_TIME)
                        break

                except Exception:
                    continue

        downloaded_count = 0
        thumbnail_index = 0

        while downloaded_count < requested_number:
            # Search for thumbnail elements again because the DOM may change.
            thumbnails = get_thumbnail_elements(driver)

            if thumbnail_index >= len(thumbnails):
                print(
                    "There are no more thumbnail images to process."
                )
                break

            thumbnail = thumbnails[thumbnail_index]
            thumbnail_index += 1

            try:
                previous_urls = {
                    url
                    for _, url in get_large_image_candidates(driver)
                }

                driver.execute_script(
                    """
                    arguments[0].scrollIntoView({
                        block: 'center',
                        inline: 'center'
                    });
                    """,
                    thumbnail,
                )

                time.sleep(0.3)

                # JavaScript clicking is less affected by overlapping elements
                # than the standard Selenium click method.
                driver.execute_script(
                    "arguments[0].click();",
                    thumbnail,
                )

                try:
                    image_url = wait_for_new_large_image(
                        driver,
                        previous_urls,
                    )

                except TimeoutException:
                    print(
                        "[Skipped] Could not find a large image URL "
                        f"for thumbnail {thumbnail_index}."
                    )
                    continue

                # Skip duplicated image URLs.
                if image_url in downloaded_urls:
                    continue

                filepath = download_image(
                    session=session,
                    url=image_url,
                    save_directory=save_directory,
                    keyword=safe_keyword,
                    number=downloaded_count + 1,
                )

                downloaded_urls.add(image_url)
                downloaded_count += 1

                print(
                    f"[{downloaded_count}/{requested_number}] "
                    f"Saved: {filepath.name}"
                )

                time.sleep(0.5)

            except StaleElementReferenceException:
                # The page structure may change after scrolling or clicking.
                continue

            except requests.RequestException as error:
                print(f"[Download failed] {error}")

            except Exception as error:
                # Do not silently ignore exceptions while debugging.
                print(
                    f"[Processing failed] Thumbnail {thumbnail_index}: "
                    f"{type(error).__name__}: {error}"
                )

        print(
            f"\nSearch completed. "
            f"{downloaded_count} images were saved.\n"
            f"Save directory: {save_directory}"
        )

    finally:
        # Close the browser and terminate the ChromeDriver process.
        driver.quit()


if __name__ == "__main__":
    main()