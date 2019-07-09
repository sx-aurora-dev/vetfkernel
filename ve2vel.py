import sys
import re
import logging
import argparse

def skip_space(src):
    text = ''
    while src:
        if src[0].isspace():
            text += src.pop(0)
        else:
            break
    return text

def isChar(c):
    return c.isalnum() or c in "#._\\"

class Token:
    def __init__(self, kind, text):
        self.kind = kind
        self.text = text

class Tokenizer:
    def __init__(self, src):
        self.src = src

    def gettoken(self):
        tmp = skip_space(self.src)
        #logging.debug("gettoken: space={}".format(len(tmp)))
        print(tmp, end="")
        if not self.src:
            return None

        c = self.src[0]
        if c in '()=;,':
            self.src.pop(0)
            token = Token(c, c) 
        elif c in '<>"{}?:*-+&%[]/!|':
            token = Token('other', self.src.pop(0))
        elif isChar(self.src[0]):
            text = ''
            while self.src:
                if isChar(self.src[0]):
                    text += self.src.pop(0)
                else:
                    break
            if '_ve_lvl' in text:
                return Token('lvl', text)
            elif '_ve_' in text:
                token = Token('intrinsic', text)
            else:
                token = Token('id', text)
        else:
            raise RuntimeError("gettoken: unknown char {}".format(self.src[0]))
        logging.debug("gettoken: return Token{{kind={}, text={}}}".format(token.kind, token.text))
        return token

class Parser:
    def __init__(self):
        self.tokenizer = None
        self.currVL = None
        self.token = None

    def gettoken(self): 
        self.token = self.tokenizer.gettoken()

    def puttoken(self):
        print(self.token.text, end="")

    def consume(self, flag = True):
        if flag:
            self.puttoken()
        self.gettoken()


    def parse_expr(self, flag = True):
        expr = ''
        while True:
            if self.token.kind in ['other', 'id']:
                expr += self.token.text
                self.consume(flag)
            elif self.token.kind in '(':
                self.consume(flag)
                self.parse_expr(flag)
                if self.token.kind != ')':
                    raise RuntimeError("unexpected token: {}".format(self.token.text))
                self.consume(flag)
            elif self.token.kind in ',)':
                break
            else:
                raise RuntimeError("unexpected token in expr: {}".format(self.token.text))
        logging.debug("parse_expr: expr={} next_token={}".format(expr, self.token.text))
        return expr

    def add_vsc_vgt_range(self, veintrin, i):
        if ('vsc' in veintrin and i == 2) or ('vgt' in veintrin and i == 1):
            print(", 0, 0", end="")

    def add_vl(self, veintrin, i):
        if hasVL(veintrin):
            if i > 0:
                print(", ", end="")
            print("{}".format(self.currVL), end="")

    def parse_intrinsic_arguments(self, veintrin):
        i = 0
        while True:
            #logging.debug("parse_list: token={}".format(token.text))
            if self.token.kind == ')':
                self.add_vsc_vgt_range(veintrin, i)
                self.add_vl(veintrin, i)
                break
            elif self.token.kind == ',':
                self.add_vsc_vgt_range(veintrin, i)
                self.consume()
            elif self.token.kind == 'id' or self.token.kind == 'other':
                self.parse_expr()
                i += 1
            elif self.token.kind == '(':
                self.consume()
                self.parse_expr()
                self.consume() # )
                i += 1
            elif self.token.kind == 'intrinsic':
                self.parse_intrin()
                i += 1
            else:
                raise RuntimeError("unknown token in intrinsic arguments: {}".format(self.token.text))
        #logging.debug("parse_list: list={} next_token={}".format(",".join(ary), token.text))

    def parse_vfmk(self, veintrin):
        self.gettoken() # skip (
        logging.debug("parse_vfmk: token={}".format(self.token.text))
        cc = self.token.text
        self.gettoken() # skip cc
        self.gettoken() # skop ,

        CC = {"VECC_G" : "gt", "VECC_L": "lt",
              "VECC_NE" : "ne", "VECC_EQ": "eq",
              "VECC_GE" : "ge", "VECC_LE": "le",
              "VECC_NUM" : "num", "VECC_NAN": "nan",
              "VECC_GNAN" : "gnan", "VECC_LNAN": "lnan",
              "VECC_NENAN" : "nenan", "VECC_EQNAN": "eqnan",
              "VECC_GENAN" : "genan", "VECC_LENAN": "lenan"}

        velintrin = re.sub(r'_ve_vfmk([dslw])_mcv', r'_vel_vfmk\1', veintrin)
        print("{}{}_mvl(".format(velintrin, CC[cc]), end="")

        self.parse_intrinsic_arguments(veintrin)

    def parse_intrin(self):
        veintrin = self.token.text
        self.gettoken()

        m = re.match(r'_ve_vfmk[dslw]', veintrin)
        if m:
            self.parse_vfmk(veintrin)
            return

        velintrin = ve2vel(veintrin)
        logging.debug("parse_intrin: {} -> {}".format(veintrin, velintrin))
        print(velintrin, end="")

        if self.token.kind == '(':
            self.consume()
            self.parse_intrinsic_arguments(veintrin)
            if self.token.kind != ')':
                raise RuntimeError("paser_intrin: expecte ')'. but {}".format(self.token.text))
            self.consume()
            #logging.debug("parse_intrin: args={}".format(", ".join(args)))
        else:
            raise RuntimeError('not (: {}'.format(self.token.text))

    def parse_lvl(self):
        if self.token.kind != '(':
            raise RuntimeError("parse_lvl: expected '(' but '{}'".format(self.token.text))
        self.gettoken()
        vl = self.parse_expr(False);
        if self.token.kind != ')':
            raise RuntimeError("parse_lvl: expected ')' but '{}'".format(self.token.text))
        self.gettoken()
        return vl

    def parse(self, text):
        self.tokenizer = Tokenizer(list(text))
        self.gettoken()
        self.currVL = None

        while self.token:
            if self.token.kind == 'id' and self.token.text == 'veintrin.h':
                print('velintrin.h', end="")
                self.gettoken()
            elif self.token.kind == 'lvl':
                self.gettoken()
                self.currVL = self.parse_lvl()
            elif self.token.kind == 'intrinsic':
                self.parse_intrin()
            else:
                self.consume()

def hasVL(intrin):
    noVL = ['lvl', 'pack', 'lvm', 'svm', 'andm', 'orm', 'xorm', 'eqvm', 'nndm', 'negm', 'svob']
    for tmp in noVL:
        if tmp in intrin:
            return False
    return True

def ve2vel(intrin):
    intrin = re.sub(r'_ve', '_vel', intrin)
    intrin = re.sub(r'vfdivsA', 'approx_vfdivs', intrin)
    intrin = re.sub(r'pvfdivA', 'approx_pvfdiv', intrin)
    intrin = re.sub(r'vbrd_vs_f64', 'vbrdd_vs', intrin)
    intrin = re.sub(r'vbrdu_vs_f32', 'vbrds_vs', intrin)
    intrin = re.sub(r'vsc(.*)_vv', r'vsc\1_vvss', intrin)
    intrin = re.sub(r'vgt(.*)_vv', r'vgt\1_vvss', intrin)
    intrin = re.sub(r'vbrd_vs_i64', 'vbrdl_vs', intrin)
    intrin = re.sub(r'vbrd_vs_i32', 'vbrdw_vs', intrin)
    intrin = re.sub(r'vbrd_vs_f64', 'vbrdd_vs', intrin)
    intrin = re.sub(r'vbrd_vs_f32', 'vbrds_vs', intrin)
    if hasVL(intrin):
        intrin += "l"
    return intrin

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", "-d", action='store_true')
    parser.add_argument("--verbose", "-v", action='store_true')
    parser.add_argument("file")
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    filename = args.file
    logging.info(filename)

    pp = Parser()

    with open(filename) as f:
        text = f.read()
        pp.parse(text)

if __name__ == "__main__":
    main()
