type automate = int * (int * int) array * bool array

let automate_2:automate = (2, [|(0, 1); (1, 0)|], [|false; true|])

let etats_accessibles (aut : automate) : int list = 
  let n, arr, f = aut in
  let rec aux rem = match rem with
    | [] -> []
    | h::t -> let a, b = arr.(h) in 
              
      