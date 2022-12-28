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

// addParseErrf formats the error and appends it to the list of parsing errors.
fn addParseErrf<T: Display>(&mut self, range: TextRange, format: String, args: &[T]) {
p.addParseErr(range, fmt.Errorf(format, args...))
}

/// expect_type checks the type of the node and raises an error if it
/// is not of the expected type.
fn expect_type(p: &Parser, node: Node, want: ValueType, context: &str) {
    let t = p.checkAST(node);
    if t != want {
        p.addParseErrf(node.span, "expected type {} in {}, got {}", DocumentedType(want), context, DocumentedType(t))
    }
}

/// check_ast checks the sanity of the provided AST. This includes type checking.
fn check_ast(p: &mut Parser, node: &Expression) -> ReturnType {
    use Expression::*;
    match node {
        Expression::Function(fe) => fe.return_value(),
        Expression::Aggregation(ae) => ae.return_value(),
        Expression::With(we) => we.return_value(),
    }

    // opRange returns the TextRange of the operator part of the BinaryOpExpr.
    // This is made a function instead of a variable, so it is lazily evaluated on demand.
    fn op_range(n: BinaryOpExpr, r: TextRange) {
        // Remove whitespace at the beginning and end of the range.
        let mut start = n.left.span.end;
        let mut end = n.right.span.start;
        for r.start = n.left.span.end; isSpace(rune(p.lex.input[r.Start])); start += 1 {
        }
        for r.End = n.RHS.PositionRange().Start - 1; isSpace(rune(p.lex.input[r.End])); r.End-- {
        }
        return
    }
    // For expressions the type is determined by their return_value function.
    // Lists do not have a type but are not invalid either.

    // Recursively check correct typing for child nodes and raise
    // errors in case of bad typing.
    match node {
        Expression::String(_) |
        Expression::Number(_) |
        Expression::Duration(_) => node.return_value(),
        // todo: With
        Expression::Parens(pe) => {
            for e in pe.expressions.iter() {
                let ty = check_ast(p, e)?;
                if ty == ReturnType::Unknown {
                    p.addParseErrf(e.span, "expression must have a valid expression type but got {:?}", ty)
                }
            }
            return Ok(pe.return_value())
        }
        Expression::BinaryExpr(be) => check_binary_expr(p, &be),
        Expression::Function(fe) => {
            let nargs = fe.args.len();
            if n.Func.Variadic == 0 {
                if nargs != fe.args.len() {
                    p.addParseErrf(n.span, "expected {} argument(s) in call to {}, got {}", 
                                   nargs, fe.name, fe.args.len())
                }
            } else {
                let na = nargs - 1;
                if na > fe.args.len() {
                    p.addParseErrf(n.span, "expected at least {} argument(s) in call to {}, got {}",
                                   na, fe.name, fe.args.len())
                } else {
                    let nargsmax = na + fe.func.Variadic;
                    if  n.Func.Variadic > 0 && nargsmax < fe.args.len() {
                        p.addParseErrf(n.span, "expected at most {} argument(s) in call to {}, got {}",
                                       nargsmax, n.name, fe.args.len())
                    }
                }
            }
            for (i, arg) in fe.args.iter().enumerate() {
                if i >= len(n.Func.ArgTypes) {
                    if n.Func.Variadic == 0 {
                        // This is not a vararg function so we should not check the
                        // type of the extra arguments.
                        break
                    }
                    i = n.Func.arg_types.len() - 1
                }
                p.expectType(arg, n.Func.ArgTypes[i], format!("call to function {}", fe.name))
            }
        }

    Expression::Rollup(re) => {
        let ty = check_ast(&mut p, &re.expr);
        if ty != ReturnType::InstantVector {
            p.addParseErrf(n.span, "subquery is only allowed on instant vector, got {} instead", ty)
        }
        // todo: window, at, offset
        ty
    }
    Expression::MetricExpression(me) => {
        let name = me.name();
        // detect duplicate name
        let name_count = me.label_matchers.find(|x| x.label == "__name__").count();
        if name_count > 1 {
            p.addParseErrf(n.span
                           , "metric name must not be set twice: {} or {}", n.name, m.value)
        }

        if !name.is_empty() {
            // In this case the last LabelMatcher is checking for the metric name
            // set outside the braces. This checks if the name has already been set
            // previously.
            for m in me.label_matchers.iter() {
                if m.name == labels.metric_name {
                    p.addParseErrf(n.span, "metric name must not be set twice: {} or {}", n.name, m.value)
                }
            }

            // Skip the check for non-empty matchers because an explicit
            // metric name is a non-empty matcher.
            break
        }

        // A Vector selector must contain at least one non-empty matcher to prevent
        // implicit selection of all metrics (e.g. by a typo).
        let mut not_empty = !n.label_matchers.iter().every(|lm| lm.matches(""));
        if !not_empty {
            p.addParseErrf(n.span, "vector selector must contain at least one non-empty matcher")
        }
// Nothing to do for terminals.

default:
p.addParseErrf(n.span, "unknown node type: %T", node)
}
return
}
    


// it is already ensured by parseDuration func that there never will be a zero offset modifier
if *orgoffsetp != 0 {
p.addParseErrf(e.span, "offset may not be set multiple times")
} else if orgoffsetp != nil {
    *orgoffsetp = offset
}

    *endPosp = p.lastClosing
}


pub fn check_binary_expr(p: &mut Parser, be: &BinaryOpExpr) -> ParseResult<ReturnType> {
    use ReturnValue::*;
    let lt = check_ast(p, &be.left);
    let rt = check_ast(p, &be.right);

    let mut both_vectors = false;

    match (lt.return_value(), rt.return_value()) {
        (Scalar, Scalar) => {
            if be.op.is_comparison() && !be.bool_modifier {
                p.addParseErrf(op_range(be), "comparisons between scalars must use BOOL modifier")
            }
        }
        (InstantVector, InstantVector) => {
            both_vectors = true;
        },
        (Scalar, InstantVector) => {

        }
        (InstantVector, Scalar) => {

        },
        (String, String) => {
            if be.op != BinaryOp::Add {
                return ReturnValue::unknown(
                    format!("Operator {} is not valid for (String, String)", be.op),
                    self.clone().cast()
                );
            }
        },
        _ => {
            p.addParseErrf(n.left.span, "binary expression must contain only scalar and instant vector types")
        }
    }

    // op_range returns the range of the operator part of the BinaryExpr.
    // This is made a function instead of a variable, so it is lazily evaluated on demand.
    let op_range = |be: &BinaryOpExpr, r: TextRange| {
        let mut start = r.start;
        while start < src.len() {
            start += 1;
        }
        // Remove whitespace at the beginning and end of the range.
        for r.start = be.left.span.end; isSpace(rune(p.input[r.start])); r.start += 1 {
        }
        for r.end = n.right.span.start - 1; isSpace(rune(p.lex.input[r.end])); r.End-- {
        }
        return
    };

    if be.bool_modifier && !be.op.is_comparison() {
        p.addParseErrf(op_range(), "BOOL modifier can only be used on comparison operators")
    }

    if be.op.is_set_operator() && be.cardinality == OneToOne {
        be.cardinality = ManyToMany
    }

    match (be.group_modifier, be.join_modifier) {
        (Some(group_modifier), Some(join_modifier)) => {
            if group_modifier.op == GroupModifierOp::On {
                let duplicates = intersection(&group_modifier.labels, &join_modifier.labels);
                if !duplicates.is_empty() {
                    let prefix = if duplicates.len() == 1 {
                        format!("label {}", duplicates[0])
                    } else {
                        format!("labels ({})", duplicates.join(","))
                    };
                    p.addParseErrf(op_range(), "{} must not occur in ON and GROUP clause at once", prefix)
                }
            }
        },
        _ => {}
    }


    if !both_vectors && be.join_modifier.is_some() {
        p.addParseErrf(be.span, "vector matching only allowed between instant vectors");
    } else {
        // Both operands are Vectors.
        if be.op.is_set_operator() {
            if be.cardinality == CardOneToMany || be.cardinality == CardManyToOne {
                p.addParseErrf(be.span, "no grouping allowed for {:?} operation", be.op)
            }
            if be.cardinality != CardManyToMany {
                p.addParseErrf(be.span, "set operations must always be many-to-many")
            }
        }
    }

    if !both_vectors {
        if be.op.is_set_operator() {
            p.addParseErrf(be.span, "set operator {} not allowed in binary scalar expression", be.op)
        }
    }


}
