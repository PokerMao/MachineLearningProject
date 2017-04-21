# Copyright 2006 Bryan O'Sullivan <bos@serpentine.com>.
#
# This software may be used and distributed according to the terms
# of the GNU General Public License, version 2 or later, which is
# incorporated herein by reference.

class SlopeOne(object):
    def __init__(self):
        self.diffs = {}
        self.freqs = {}

    def predict(self, userprefs):
        preds, freqs = {}, {}
        for item, rating in userprefs.iteritems():
            for diffitem, diffratings in self.diffs.iteritems():
                try:
                    freq = self.freqs[diffitem][item]
                except KeyError:
                    continue
                preds.setdefault(diffitem, 0.0)
                freqs.setdefault(diffitem, 0)
                preds[diffitem] += freq * (diffratings[item] + rating)
                freqs[diffitem] += freq
        return dict([(item, value / freqs[item])
                     for item, value in preds.iteritems()
                     if item not in userprefs and freqs[item] > 0])

    def update(self, userdata):
        for ratings in userdata.itervalues():
            for item1, rating1 in ratings.iteritems():
                self.freqs.setdefault(item1, {})
                self.diffs.setdefault(item1, {})
                for item2, rating2 in ratings.iteritems():
                    self.freqs[item1].setdefault(item2, 0)
                    self.diffs[item1].setdefault(item2, 0.0)
                    self.freqs[item1][item2] += 1
                    self.diffs[item1][item2] += rating1 - rating2
        for item1, ratings in self.diffs.iteritems():
            for item2 in ratings:
                ratings[item2] /= self.freqs[item1][item2]

if __name__ == '__main__':
    userdata = {}
    f = open("small_train_data.csv", 'r')
    next(f)
    print("miao1")
    for line in f:
        data = line.strip().split(',')
        userId = data[1]
        movieId = data[2]
        rating = float(data[3])
        if userId in userdata:
            userdata[userId][movieId] = rating
        else:
            userdata[userId] = {}
            userdata[userId][movieId] = rating
    s = SlopeOne()
    s.update(userdata)