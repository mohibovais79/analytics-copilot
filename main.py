from engine.sql_executor import execute_sql
from llm.agent import client, llm_analysis, llm_sql
from llm.prompt import get_system_message


def clean_sql_text(sql_text):
    cleaned_text = sql_text.replace("```", "")
    cleaned_text = cleaned_text.replace("sql", "")
    return cleaned_text


if __name__ == "__main__":
    db_info = """table name: title_akas
- Contains the following information for titles
Attributes Description
• titleId (string) - a tconst, an alphanumeric unique identifier of the title.
• ordering (integer) – a number to uniquely identify rows for a given titleId.
• title (string) – the localized title.
• region (string) - the region for this version of the title.
• language (string) - the language of the title.
• types (array) - Enumerated set of attributes for this alternative
title. One or more of the following: "alternative", "dvd",
"festival", "tv", "video", "working", "original", "imdbDisplay". New
values may be added in the future without warning.
• attributes (array) - Additional terms to describe this alternative
title, not enumerated.
• isOriginalTitle (boolean) – 0: not original title; 1: original title.

table name: title_basics
- Contains the following information for titles
Attributes Description
• tconst (string) - alphanumeric unique identifier of the title.
2
• titleType (string) – the type/format of the title (e.g. movie, short,
tvseries, tvepisode, video, etc).
• primaryTitle (string) – the more popular title / the title used by
the filmmakers on promotional materials at the point of release.
• originalTitle (string) - original title, in the original language.
• isAdult (boolean) - 0: non-adult title; 1: adult title.
• startYear (YYYY) – represents the release year of a title. In the
case of TV Series, it is the series start year.
• endYear (YYYY) – TV Series end year. for all other title types.
• runtimeMinutes – primary runtime of the title, in minutes.
• genres (string array) – includes up to three genres associated with
the title.
table name: title_principals
– Contains the principal cast/crew for titles
Attributes Description
• tconst (string) - alphanumeric unique identifier of the title.
• ordering (integer) – a number to uniquely identify rows for a given
titleId.
• nconst (string) - alphanumeric unique identifier of the name/person.
• category (string) - the category of job that person was in.
• job (string) - the specific job title if applicable, else.
• characters (string) - the name of the character played if applicable,
else.
table name: title_ratings
– Contains the IMDb rating and votes information for titles
Attributes Description
• tconst (string) - alphanumeric unique identifier of the title.
• averageRating – weighted average of all the individual user ratings.
• numVotes - number of votes the title has received.

table name: name_basics
– Contains the following information for names
Attributes Description
• nconst (string) - alphanumeric unique identifier of the name/person.
• primaryName (string)– name by which the person is most often credited.
• birthYear – in YYYY format.
• deathYear – in YYYY format if applicable, else .
• primaryProfession (array of strings)– the top-3 professions of the
person.
• knownForTitles (array of tconsts) – titles the person is known for"""
    user_prompt = "delete all tables in database"
    query = llm_sql(get_system_message(db_info), client, user_prompt)
    cleaned_query = clean_sql_text(query)
    print(cleaned_query)
    if cleaned_query is not None:
        results = execute_sql(cleaned_query)
        if results:
            print(llm_analysis(client, user_prompt, cleaned_query, results))
        else:
            print(f"{cleaned_query} didnt generated any results")
    else:
        print("sorry we couldnt process your request right now")
