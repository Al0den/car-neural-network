type elt = char
type tree = E | N of tree * elt * tree

let exemple : tree = N(N(N(E, 'd', E), 'b', E), 'a', N(N(E, 'e', E), 'c', E))

let rec get (a : int) (t : tree) : elt = match a, t with
  | 0, N(_, e, _) -> e
  | _, N(g, _, d) -> (match a mod 2 with
      | 0 -> get (a/2 - 1) d
      | 1 -> get ((a-1)/2) g
      | _ -> failwith "Impossible 2")
  | _ -> failwith "impossible 1"

let _ = print_string("Question 2.2")
let _ = print_newline()
let _ = print_char(get 3 exemple)
(*
On a n valeurs, qui se repartisse en 2^k au k-eme etage. On a donc bien un arbre de hauteur en O(log(n))
*)

let rec liat n = function 
  | N(E, _, E) -> E
  | N(l, x, r) -> if n mod 2 = 0 then N(liat (n/2) l, x, r)
                                else N(l, x, liat (n/2) r)
  | E -> assert false
(*
Elle est en O(log(n)), on divise n par deux a chaque iteration
*)

(*
L'arbre etant finit, et comme on le parcours en descendant a travers le-dit arbre, on fait moins de n appels a la fonction, 
donc elle termine.

On montre avec un invariant que l'appel se dirige bien vers l'element de plus grand indice
Enfin, c'est cet element qui est enlever, et le reste de l'arbre est preserver
*)
let rec snoc (n : int) (t : tree) (x : elt) : tree = match n, t with
  | 0, E -> N(E, x, E)
  | _, N(g, e, d) -> if n mod 2 = 0 then N(g, e, snoc (n/2 - 1) d x)
                                    else N(snoc ((n-1)/2) g x, e, d)
  | _ -> failwith "Impossible 3"

let rec tail (t : tree) : tree = match t with
  | N(E, _, E) -> E
  | N(l, x, r) -> if l = E then r
                          else N(tail l, x, r)
  | E -> assert false

let cons (a : elt) (t : tree) (n : int) : tree = 
  let rec cons_aux (t : tree) (n : int) : tree = match t with
    | E -> N(E, a, E)
    | N(l, x, r) -> if n mod 2 = 0 then N(cons_aux l (n/2 - 1), x, r)
                                    else N(l, x, cons_aux r ((n-1)/2))
  in cons_aux t n

let of_array (a : elt array) : tree = 
  let rec of_array_aux (a : elt array) (n : int) : tree = match n with
    | 0 -> E
    | _ -> snoc (n-1) (of_array_aux a (n/2)) a.(n-1)
  in of_array_aux a (Array.length a)

let make (n : int) (x : elt) : tree = 
  let rec make_aux (n : int) : tree = match n with
    | 0 -> E
    | _ -> N(make_aux (n/2), x, make_aux (n/2))
  in make_aux n

let exemple_2 : tree = snoc 5 exemple 'x'
let exemple_3 : tree = make 4 'a'
let _ = print_string("Test")
let _ = print_char(get 4 exemple_3)


let _ = print_newline()
let _ = print_string("Question 2.6")
let _ = print_newline()
let _ = print_char(get 5 exemple_2)

