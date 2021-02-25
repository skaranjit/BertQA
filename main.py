
from absl import app,flags,logging
import sh
import create_dataset
from loadModel import *


flags.DEFINE_string('model','bert-base-uncased','')
flags.DEFINE_string('pdf',None,'Input file name.', short_name='pd')
flags.DEFINE_string('txt',None,'Input file name.', short_name='txt')
flags.DEFINE_string('article',None,'Input file name.', short_name='ar')

FLAGS=flags.FLAGS


def main(_):
    p = create_dataset.Create_DS()
    if(FLAGS.pdf != None):
        logging.info("Hello")
        p.loadPdf()
    if(FLAGS.txt != None):
        logging.info("In Text ")
        p.loadTxt()
    if(FLAGS.article != None):
        logging.info("Internet!!")
        p.loadArticle(FLAGS.article)
        print(p.ds)

    model = QAPipe(p.ds)
    result = model.get_output("What happens after life")
    import IPython; IPython.embed(); exit(1);

if __name__ == '__main__':
    app.run(main)