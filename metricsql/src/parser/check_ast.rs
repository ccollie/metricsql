// Copyright 2015 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


use std::fmt::{Display, Formatter};
use std::string::ParseError;

struct parser {
    lex: Lexer,

    inject:    ItemType,
    injecting: bool,

    // Everytime an Item is lexed that could be the end
    // of certain expressions its end position is stored here.
    last_closing: Pos,

    generatedParserResult: interface,
    parse_errors:          ParseErrors
};



type ParseErrors = Vec<ParseErr>;

// ParseExpr returns the expression parsed from the input.
fn ParseExpr(input: string) -> Result<Expression> {
    let p = newParser(input)
    parseResult := p.parseGenerated(START_EXPRESSION)
    if parseResult != nil {
        expr = parseResult.(Expr)
    }
    p.checkAST(expr);

    if len(p.parseErrors) != 0 {
        err = p.parseErrors
    }

    return expr
}

// SequenceValue is an omittable value in a sequence of time series values.
struct SequenceValue {
    value: f64,
    omitted: bool
}


struct SeriesDescription {
    labels: Labels,
    values: Vec<SequenceValue>
}

// addParseErrf formats the error and appends it to the list of parsing errors.
fn addParseErrf<T: Display>(&mut self, range: TextRange, format: String, args: &[T]) {
p.addParseErr(range, fmt.Errorf(format, args...))
}

// addParseErr appends the provided error to the list of parsing errors.
func (p *parser) addParseErr(range: TextRange, err: ParseError) {
    let perr = ParseErr{
        range: range,
        err,
        query:         p.lex.input,
    }
    p.parseErrors = append(p.parseErrors, perr)
}



var errUnexpected = errors.New("unexpected error")


// expectType checks the type of the node and raises an error if it
// is not of the expected type.
fn expectType(p: &Parser, node: Node, want: ValueType, context: string) {
    let t = p.checkAST(node);
    if t != want {
        p.addParseErrf(node.range(), "expected type {} in {}, got {}", DocumentedType(want), context, DocumentedType(t))
    }
}

// checkAST checks the sanity of the provided AST. This includes type checking.
fn checkAST(p: &Parser, node: Node) -> ValueType {
    // For expressions the type is determined by their Type function.
    // Lists do not have a type but are not invalid either.
    match n := node.(type) {
        case Expressions:
            typ = ValueTypeNone
            case Expr:
typ = n.Type()
default:
p.addParseErrf(node.range(), "unknown node type: %T", node)
}

// Recursively check correct typing for child nodes and raise
// errors in case of bad typing.
    match n {
        case *EvalStmt:
        let ty = p.checkAST(n.Expr)
        if ty == ValueTypeNone {
            p.addParseErrf(n.Expr.range(), "evaluation statement must have a valid expression type but got %s", DocumentedType(ty))
        }

        Expression::Parens(p) => {
            for e in p.expressions.iter() {
                let ty = p.checkAST(e)
                if ty == ValueTypeNone {
                    p.addParseErrf(e.range(), "expression must have a valid expression type but got {}", DocumentedType(ty))
                }
            }
        }
        Expression::BinaryExpr(be) => {
            let lt = p.checkAST(n.left);
            let rt = p.checkAST(n.right);

            if (lt != Scalar && lt != InstantVector) ||
                (rt != Scalar && rt != InstantVector) {
                p.addParseErrf(n.left.range(), "binary expression must contain only scalar and instant vector types")
            }

            // opRange returns the range of the operator part of the BinaryExpr.
            // This is made a function instead of a variable, so it is lazily evaluated on demand.
            let opRange = |be: &BinaryOpExpr, r: TextRange| {
                let mut start = r.start;
                while start < src.len() {
                    start += 1;
                }
                // Remove whitespace at the beginning and end of the range.
                for r.start = be.left.span.end; isSpace(rune(p.lex.input[r.Start])); r.start += 1 {
                }
                for r.end = n.right.span.start - 1; isSpace(rune(p.lex.input[r.End])); r.End-- {
                }
                return
            };

            if be.bool_modifier && !be.op.is_comparison() {
                p.addParseErrf(opRange(), "bool modifier can only be used on comparison operators")
            }

            if be.op.is_comparison() && !be.bool_modifier && rt == Scalar && lt == Scalar {
                p.addParseErrf(opRange(), "comparisons between scalars must use BOOL modifier")
            }

            if be.op.is_set_operator() && be.cardinality == OneToOne {
                be.cardinality = ManyToMany
            }

            match (self.group_modifier, self.label_modifier) {
                (Some(group_modifier), Some(label_modifier)) => {
                    if group_modifier.op == GroupModifierOp::On {
                        let duplicates = intersection(&group_modifier.labels, &label_modifier.labels);
                        if !duplicates.is_empty() {
                            p.addParseErrf(opRange(), "labels {} must not occur in ON and GROUP clause at once", duplicates.join(", "))
                        }
                    }
                },
                _ => {
                    //
                }
            }


            if (lt != InstantVector || rt != InstantVector) && be.join_modifier.is_some() {
                p.addParseErrf(be.range(), "vector matching only allowed between instant vectors");
            } else {
                // Both operands are Vectors.
                if be.op.is_set_operator() {
                    if be.VectorMatching.Card == CardOneToMany || be.VectorMatching.Card == CardManyToOne {
                        p.addParseErrf(be.range(), "no grouping allowed for {} operation", be.op)
                    }
                    if be.cardinality != CardManyToMany {
                        p.addParseErrf(be.range(), "set operations must always be many-to-many")
                    }
                }
            }

            if (lt == Scalar || rt == Scalar) && be.op.is_set_operator() {
                p.addParseErrf(be.range(), "set operator {} not allowed in binary scalar expression", n.op)
            }
        }

case *Call:
nargs := len(n.Func.ArgTypes)
if n.Func.Variadic == 0 {
if nargs != len(n.Args) {
p.addParseErrf(n.range(), "expected %d argument(s) in call to %q, got %d", nargs, n.Func.Name, len(n.Args))
}
} else {
na := nargs - 1
if na > len(n.Args) {
p.addParseErrf(n.range(), "expected at least %d argument(s) in call to %q, got %d", na, n.Func.Name, len(n.Args))
} else if nargsmax := na + n.Func.Variadic; n.Func.Variadic > 0 && nargsmax < len(n.Args) {
p.addParseErrf(n.range(), "expected at most %d argument(s) in call to %q, got %d", nargsmax, n.Func.Name, len(n.Args))
}
}

for i, arg := range n.Args {
if i >= len(n.Func.ArgTypes) {
if n.Func.Variadic == 0 {
// This is not a vararg function so we should not check the
// type of the extra arguments.
break
}
i = len(n.Func.ArgTypes) - 1
}
p.expectType(arg, n.Func.ArgTypes[i], fmt.Sprintf("call to function %q", n.Func.Name))
}

case *ParenExpr:
p.checkAST(n.Expr)

case *UnaryExpr:
        if n.op != ADD && n.op != SUB {
            p.addParseErrf(n.range(), "only + and - operators allowed for unary expressions")
        }
        if t := p.checkAST(n.Expr); t != ValueTypeScalar && t != ValueTypeVector {
            p.addParseErrf(n.range(), "unary expression only allowed on expressions of type scalar or instant vector, got %q", DocumentedType(t))
        }

case *SubqueryExpr:
ty := p.checkAST(n.Expr)
if ty != ValueTypeVector {
p.addParseErrf(n.range(), "subquery is only allowed on instant vector, got %s instead", ty)
}
case *MatrixSelector:
p.checkAST(n.VectorSelector)

    Expression::MetricExpression(me) => {
        let name = me.name();
        // detect duplicate name
        let name_count = me.label_matchers.find(|x| x.label == "__name__").count();
        if name_count > 1 {
            p.addParseErrf(n.range(), "metric name must not be set twice: {} or {}", n.name, m.value)
        }

        if !name.is_empty() {
            // In this case the last LabelMatcher is checking for the metric name
            // set outside the braces. This checks if the name has already been set
            // previously.
            for m in me.label_matchers.iter() {
                if m.Name == labels.MetricName {
                    p.addParseErrf(n.range(), "metric name must not be set twice: {} or {}", n.name, m.value)
                }
            }

            // Skip the check for non-empty matchers because an explicit
            // metric name is a non-empty matcher.
            break
        }

        // A Vector selector must contain at least one non-empty matcher to prevent
        // implicit selection of all metrics (e.g. by a typo).
        let mut not_empty = false;
        for lm in n.label_matchers.iter() {
            if lm != nil && !lm.Matches("") {
                not_empty = true;
                break
            }
        }
        if !not_empty {
            p.addParseErrf(n.range(), "vector selector must contain at least one non-empty matcher")
        }

    Number | String  => {}
// Nothing to do for terminals.

default:
p.addParseErrf(n.range(), "unknown node type: %T", node)
}
return
}
    


// it is already ensured by parseDuration func that there never will be a zero offset modifier
if *orgoffsetp != 0 {
p.addParseErrf(e.range(), "offset may not be set multiple times")
} else if orgoffsetp != nil {
*orgoffsetp = offset
}

*endPosp = p.lastClosing
}


pub fn intersection(labels_a: &Vec<String>, labels_b: &Vec<String>) -> Vec<String> {
    if labels_a.is_empty() || labels_b.is_empty() {
        return vec![]
    }
    let unique_a: HashSet<String> = labels_a.clone().into_iter().collect();
    let unique_b: HashSet<String> = labels_b.clone().into_iter().collect();
    unique_a
        .intersection(&unique_b)
        .map(|i| *i)
        .collect::<Vec<_>>()
}
