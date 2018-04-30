import readability

def get_scores(df):
    reviews = df
    read_scores = []
    text_scores = []
    for review in reviews:
        rd = readability.Readability(review)
        text_scores = [rd.ARI(), rd.FleschReadingEase(), rd.FleschKincaidGradeLevel(), rd.GunningFogIndex(), rd.SMOGIndex(), rd.ColemanLiauIndex(), rd.LIX(), rd.RIX()]
        read_scores.append(text_scores)
    return read_scores
