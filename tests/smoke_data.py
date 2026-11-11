"""Синтетический корпус для тестов: 4 темы, дисбаланс 45/30/17/8 %."""
import random

TOPICS = {
    'спорт': 'матч гол команда тренер чемпионат игрок счёт турнир лига победа',
    'экономика': 'рубль банк инфляция рынок акции нефть бюджет кредит ставка цены',
    'наука': 'учёные исследование космос телескоп молекула эксперимент открытие клетка физика вирус',
    'культура': 'фильм театр премьера режиссёр выставка музей концерт книга актёр роман',
}
WEIGHTS = {'спорт': 0.45, 'экономика': 0.3, 'наука': 0.17, 'культура': 0.08}
COMMON = 'в на и что по с не это год как также после был для было его который'.split()


def make_rows(n=1500, seed=0):
    rng = random.Random(seed)
    labels = list(WEIGHTS)
    texts, out = [], []
    for _ in range(n):
        lab = rng.choices(labels, [WEIGHTS[l] for l in labels])[0]
        own = TOPICS[lab].split()
        other = TOPICS[rng.choice(labels)].split()
        words = [rng.choice(own) if rng.random() < .35 else rng.choice(other) if rng.random() < .3 else rng.choice(COMMON)
                 for _ in range(rng.randint(8, 40))]
        texts.append(' '.join(words))
        out.append(lab)
    return texts, out
