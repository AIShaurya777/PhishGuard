"""Where each of the dataset's 50 numeric columns comes from.

The groups follow the PhiUSIIL paper's description of its features (decision D9).
"""

# Computed from the URL string itself.
URL_COLUMNS = [
    "URLLength", "DomainLength", "IsDomainIP", "CharContinuationRate", "TLDLength",
    "NoOfSubDomain", "HasObfuscation", "NoOfObfuscatedChar", "ObfuscationRatio",
    "NoOfLettersInURL", "LetterRatioInURL", "NoOfDegitsInURL", "DegitRatioInURL",
    "NoOfEqualsInURL", "NoOfQMarkInURL", "NoOfAmpersandInURL",
    "NoOfOtherSpecialCharsInURL", "SpacialCharRatioInURL", "IsHTTPS",
]

# Scores computed against outside reference data: a list of legitimate URLs
# (URLSimilarityIndex) or probability tables for TLDs and characters.
STATISTIC_COLUMNS = ["URLSimilarityIndex", "TLDLegitimateProb", "URLCharProb"]

# Computed from the downloaded HTML page, which a URL-only app can't get.
PAGE_COLUMNS = [
    "LineOfCode", "LargestLineLength", "HasTitle", "DomainTitleMatchScore",
    "URLTitleMatchScore", "HasFavicon", "Robots", "IsResponsive", "NoOfURLRedirect",
    "NoOfSelfRedirect", "HasDescription", "NoOfPopup", "NoOfiFrame",
    "HasExternalFormSubmit", "HasSocialNet", "HasSubmitButton", "HasHiddenFields",
    "HasPasswordField", "Bank", "Pay", "Crypto", "HasCopyrightInfo", "NoOfImage",
    "NoOfCSS", "NoOfJS", "NoOfSelfRef", "NoOfEmptyRef", "NoOfExternalRef",
]

COLUMN_GROUPS = {"URL": URL_COLUMNS, "statistic": STATISTIC_COLUMNS, "page": PAGE_COLUMNS}
