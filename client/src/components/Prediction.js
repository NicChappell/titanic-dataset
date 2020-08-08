// dependencies
import React from 'react'

const Prediction = ({ prediction }) => {
    const response = prediction => {
        switch (prediction) {
            case 1:
                return (
                    <div className="positive center-align">
                        <i className="material-icons">sentiment_very_satisfied</i>
                        <p>You made it into a lifeboat!</p>
                    </div>
                )
            case 0:
                return (
                    <div className="negative center-align">
                        <i className="material-icons">sentiment_very_dissatisfied</i>
                        <p>You were left behind.</p>
                    </div>
                )
            default:
                return (
                    <div className="undefined center-align">
                        <p>Submit your information to see your prediction.</p>
                    </div>
                )
        }
    }

    return (
        <div className="row">
            <div className="col s12 l8">
                <div className="intro">
                    <h1>Would you survive?</h1>
                    <p>On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank
                    after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone onboard,
                    resulting in the death of 1,502 out of 2,224 passengers and crew.</p>
                    <p>While there was some element of luck involved in surviving, it seems some groups of people were more
                    likely to survive than others. Enter your profile data and select your travel options to see if you would have survived the sinking
                    of the Titanic.</p>
                </div>
            </div>
            <div className="col s12 l4">
                <div className="prediction">
                    {response(prediction)}
                </div>
            </div>
        </div>
    )
}

export default Prediction
