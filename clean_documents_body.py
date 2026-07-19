import re
import unicodedata
from bs4 import BeautifulSoup, NavigableString, Tag
import re
import nltk

from pymorphy3 import MorphAnalyzer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer


nltk.download("stopwords")





class TFIDFPreprocessor:
    def __init__(self, keep_urls: bool = False):
        self.parser = RAGHtmlParser(keep_urls=keep_urls)
        self.morph = MorphAnalyzer()
        self.russian_stopwords = set(stopwords.words("russian")) 

    def preprocess(self, html: str) -> str:
        html_parsed =  self.parser.parse(html)
        return self.preprocess_text_for_idf(html_parsed)

    def preprocess_text_for_idf(self, text: str) -> str:
        # lowercase
        text = text.lower()

        # оставляем только буквы
        text = re.sub(r"[^а-яё\s]", " ", text)

        # удаляем лишние пробелы
        text = re.sub(r"\s+", " ", text).strip()

        # токенизация
        tokens = text.split()

        processed_tokens = []

        for token in tokens:
            # удаляем короткие слова
            if len(token) < 2:
                continue

            # удаляем стоп-слова
            if token in self.russian_stopwords:
                continue

            # лемматизация
            lemma = self.morph.parse(token)[0].normal_form

            processed_tokens.append(lemma)

        return " ".join(processed_tokens)


class RAGHtmlParser:
    def __init__(self, keep_urls: bool = False):
        self.keep_urls = keep_urls

    def parse(self, html: str) -> str:
        soup = BeautifulSoup(html, "lxml")

        # Удаляем полностью бесполезные элементы
        for tag in soup([
            "script",
            "style",
            "noscript",
            "iframe",
            "svg",
            "canvas"
        ]):
            tag.decompose()

        text = self._parse_children(soup)

        text = self._remove_emoji(text)
        text = text.replace("\xa0", " ")

        # Лишние пробелы
        text = re.sub(r"[ \t]+", " ", text)

        # Более двух пустых строк подряд -> две
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    def _parse_children(self, node) -> str:
        result = []

        for child in node.children:

            if isinstance(child, NavigableString):
                txt = str(child)
                if txt.strip():
                    result.append(txt)
                continue

            if not isinstance(child, Tag):
                continue

            name = child.name.lower()

            # ---------- Заголовки ----------
            if name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
                level = int(name[1])
                result.append(
                    "\n"
                    + "#" * level
                    + " "
                    + self._parse_children(child).strip()
                    + "\n"
                )

            # ---------- Абзацы ----------
            elif name == "p":
                result.append(
                    self._parse_children(child).strip()
                    + "\n"
                )

            # ---------- Списки ----------
            elif name == "ul":
                result.append(self._parse_ul(child))

            elif name == "ol":
                result.append(self._parse_ol(child))

            # ---------- Таблицы ----------
            elif name == "table":
                result.append(self._parse_table(child))

            # ---------- Ссылки ----------
            elif name == "a":
                text = child.get_text(" ", strip=True)

                if self.keep_urls and child.get("href"):
                    result.append(f"{text} ({child['href']})")
                else:
                    result.append(text)

            # ---------- Предупреждения ----------
            elif "factoid" in child.get("class", []):

                label = self._detect_factoid(child)

                block = self._parse_children(child).strip()

                result.append(f"\n[{label}]\n{block}\n")

            # ---------- Спойлеры ----------
            elif (
                "spoiler" in child.get("class", [])
                or child.get("data-type") == "spoiler"
            ):
                result.append(
                    "\n"
                    + self._parse_children(child).strip()
                    + "\n"
                )

            # ---------- Перенос строки ----------
            elif name == "br":
                result.append("\n")

            # ---------- Всё остальное ----------
            else:
                result.append(self._parse_children(child))

        return "".join(result)

    def _parse_ul(self, ul: Tag) -> str:
        out = []

        for li in ul.find_all("li", recursive=False):
            text = self._parse_children(li).strip()
            out.append(f"- {text}")

        return "\n".join(out) + "\n"

    def _parse_ol(self, ol: Tag) -> str:
        out = []

        start = int(ol.get("start", 1))

        for i, li in enumerate(
            ol.find_all("li", recursive=False),
            start=start
        ):
            text = self._parse_children(li).strip()
            out.append(f"{i}. {text}")

        return "\n".join(out) + "\n"

    def _parse_table(self, table: Tag) -> str:
        rows = []

        headers = [
            c.get_text(" ", strip=True)
            for c in table.find_all("th")
        ]

        for tr in table.find_all("tr"):

            cells = [
                c.get_text(" ", strip=True)
                for c in tr.find_all(["td", "th"])
            ]

            if not cells:
                continue

            if headers and len(headers) == len(cells):
                row = []

                for h, v in zip(headers, cells):
                    row.append(f"{h}: {v}")

                rows.append("; ".join(row))

            else:
                rows.append(", ".join(cells))

        return "\n".join(rows) + "\n"

    def _detect_factoid(self, tag: Tag) -> str:

        classes = set(tag.get("class", []))

        if "factoid_warning" in classes:
            return "ВНИМАНИЕ"

        if "factoid_tip" in classes:
            return "СОВЕТ"

        if "factoid_success" in classes:
            return "ВАЖНО"

        if "factoid_disclaimer" in classes:
            return "ПРИМЕЧАНИЕ"

        return "ИНФОРМАЦИЯ"

    @staticmethod
    def _remove_emoji(text: str) -> str:
        return "".join(
            ch
            for ch in text
            if unicodedata.category(ch) != "So"
        )