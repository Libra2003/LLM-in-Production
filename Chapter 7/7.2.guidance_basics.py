from guidance import gen, select, models

falcon = models.Transformers("tiiuae/falcon-rw-1b")

lm = falcon + "Once upon a time, " + gen(max_tokens=10)
print(lm)

lm= (
    falcon
    + "write a sentence about prining press. "
    + gen(stop=["\n", ".","!"])
)
print(lm)

lm = falcon + "I like the color " + select(["cyan", "grey", "purple"])
print(lm)

lm = falcon + "Generate an email: " + gen(regex="\w+@w+.com")
print(lm)